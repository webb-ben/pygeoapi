# =================================================================
#
# Authors: Benjamin Webb <benjamin.miller.webb@gmail.com>
# Authors: Tom Kralidis <tomkralidis@gmail.com>
#
# Copyright (c) 2025 Benjamin Webb
# Copyright (c) 2022 Tom Kralidis
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# =================================================================

from json.decoder import JSONDecodeError
import logging
from requests import Session
from requests.exceptions import ConnectionError
from urllib.parse import urlparse

from pygeoapi.config import get_config
from pygeoapi.crs import get_transform_from_spec, crs_transform_feature
from pygeoapi.provider.base import (
    BaseProvider,
    ProviderQueryError,
    ProviderConnectionError,
    ProviderInvalidDataError
)
from pygeoapi.util import (
    format_datetime as fmtd,
    url_join,
    get_provider_default,
    get_base_url,
    get_typed_value
)

LOGGER = logging.getLogger(__name__)

ENTITY = {
    'Thing',
    'Things',
    'Observation',
    'Observations',
    'Location',
    'Locations',
    'Sensor',
    'Sensors',
    'Datastream',
    'Datastreams',
    'ObservedProperty',
    'ObservedProperties',
    'FeatureOfInterest',
    'FeaturesOfInterest',
    'HistoricalLocation',
    'HistoricalLocations'
}
EXPAND = {
    'Things': 'Locations,Datastreams',
    'Observations': 'Datastream,FeatureOfInterest',
    'ObservedProperties': 'Datastreams/Thing/Locations',
    'Datastreams': """
        Sensor
        ,ObservedProperty
        ,Thing/Locations
        ,Observations(
            $orderby=phenomenonTime_desc
            )
        ,Observations/FeatureOfInterest(
            $select=feature
            )
    """
}


class SensorThingsProvider(BaseProvider):
    """SensorThings API (STA) Provider"""

    # Remove whitespace and replace underscores with spaces
    expand = {
        k: ''.join(v.split()).replace('_', ' ') for (k, v) in EXPAND.items()
    }

    def __init__(self, provider_def):
        """
        STA Class constructor

        :param provider_def: provider definitions from yml pygeoapi-config.
                             data,id_field, name set in parent class

        :returns: pygeoapi.provider.sensorthings.SensorThingsProvider
        """

        LOGGER.debug('Initializing SensorThings provider')
        self.linked_entity = {}
        super().__init__(provider_def)

        # Start session
        self.http = Session()
        self._generate_mappings(provider_def)
        LOGGER.debug(f'STA endpoint: {self.data}, Entity: {self.entity}')
        self.get_fields()

    def get_fields(self):
        """
        Get fields of STA Provider

        :returns: dict of fields
        """

        if not self._fields:
            try:
                r = self._get_response(params={'$top': 1})
                fields = r['value'][0]

            except IndexError:
                LOGGER.warning('could not get fields; returning empty set')
                return {}

            except (ConnectionError, ProviderConnectionError):
                msg = f'Unable to contact SensorThings endpoint at {self._url}'
                LOGGER.error(msg)
                raise ProviderConnectionError(msg)

            for fieldname, fieldinfo in fields.items():
                iscomplex = isinstance(fieldinfo, (dict, list))
                isentity = fieldname in ENTITY

                if isinstance(fieldinfo, (int, float)):
                    self._fields[fieldname] = {'type': 'number'}

                elif isinstance(fieldinfo, str) or (iscomplex and isentity):
                    self._fields[fieldname] = {'type': 'string'}

        return self._fields

    def query(
        self,
        offset=0,
        limit=10,
        resulttype='results',
        bbox=[],
        datetime_=None,
        properties=[],
        sortby=[],
        select_properties=[],
        skip_geometry=False,
        crs_transform_spec=None,
        q=None,
        **kwargs
    ):
        """
        STA query

        :param offset: starting record to return (default 0)
        :param limit: number of records to return (default 10)
        :param resulttype: return results or hit limit (default results)
        :param bbox: bounding box [minx,miny,maxx,maxy]
        :param datetime_: temporal (datestamp or extent)
        :param properties: list of tuples (name, value)
        :param sortby: list of dicts (property, order)
        :param select_properties: list of property names
        :param skip_geometry: bool of whether to skip geometry (default False)
        :param crs_transform_spec: `CrsTransformSpec` instance, optional
        :param q: full-text search term(s)

        :returns: dict of GeoJSON FeatureCollection
        """

        return self._load(
            offset,
            limit,
            resulttype,
            bbox=bbox,
            datetime_=datetime_,
            properties=properties,
            sortby=sortby,
            select_properties=select_properties,
            crs_transform_spec=crs_transform_spec,
            skip_geometry=skip_geometry
        )

    def get(self, identifier, crs_transform_spec=None, **kwargs):
        """
        Query STA by id

        :param identifier: feature id
        :param crs_transform_spec: `CrsTransformSpec` instance, optional

        :returns: dict of single GeoJSON feature
        """

        id = self.to_iotid(identifier)
        response = self._get_response(f'{self._url}({id})')
        crs_transform_func = get_transform_from_spec(crs_transform_spec)

        return self._make_feature(
            response, crs_transform_func=crs_transform_func
        )

    def create(self, item):
        """
        Create a new item

        :param item: `dict` of new item

        :returns: identifier of created item
        """

        response = self.http.post(self._url, json=item)

        if response.status_code == 201:
            location = response.headers.get('Location')
            iotid = location[location.find('('):location.find(')')][1:]

            LOGGER.debug(f'Feature created with @iot.id: {iotid}')
            return get_typed_value(iotid)

        else:
            msg = f'Failed to create item: {response.text}'
            raise ProviderInvalidDataError(msg)

    def update(self, identifier, item):
        """
        Updates an existing item

        :param identifier: feature id
        :param item: `dict` of partial or full item

        :returns: `bool` of update result
        """

        id = self.to_iotid(identifier)
        LOGGER.debug(f'Updating @iot.id: {id}')
        response = self.http.put(f'{self._url}({id})', json=item)

        if response.status_code == 200:
            return True
        else:
            msg = f'Failed to update item: {response.text}'
            raise ProviderConnectionError(msg)

    def delete(self, identifier):
        """
        Deletes an existing item

        :param identifier: item id

        :returns: `bool` of deletion result
        """

        id = self.to_iotid(identifier)
        LOGGER.debug(f'Deleting @iot.id: {id}')
        response = self.http.delete(f'{self._url}({id})')

        if response.status_code == 200:
            return True
        else:
            msg = f'Failed to delete item: {response.text}'
            raise ProviderConnectionError(msg)

    def _load(
        self,
        offset=0,
        limit=10,
        resulttype='results',
        bbox=[],
        datetime_=None,
        properties=[],
        sortby=[],
        select_properties=[],
        skip_geometry=False,
        crs_transform_spec=None,
        q=None
    ):
        """
        Private function: Load STA data

        :param offset: starting record to return (default 0)
        :param limit: number of records to return (default 10)
        :param resulttype: return results or hit limit (default results)
        :param bbox: bounding box [minx,miny,maxx,maxy]
        :param datetime_: temporal (datestamp or extent)
        :param properties: list of tuples (name, value)
        :param sortby: list of dicts (property, order)
        :param select_properties: list of property names
        :param skip_geometry: bool of whether to skip geometry (default False)
        :param crs_transform_spec: `CrsTransformSpec` instance, optional
        :param q: full-text search term(s)

        :returns: dict of GeoJSON FeatureCollection
        """

        # Make defaults
        fc = {'type': 'FeatureCollection', 'features': []}
        params = {'$skip': str(offset), '$top': str(limit)}

        if properties or bbox or datetime_:
            params['$filter'] = self._make_filter(properties, bbox, datetime_)

        if sortby:
            params['$orderby'] = self._make_orderby(sortby)

        # Send request
        LOGGER.debug('Sending query')
        if resulttype == 'hits':
            LOGGER.debug('Returning hits')
            params['$count'] = 'true'
            response = self._get_response(params=params)
            fc['numberMatched'] = response.get('@iot.count')
            return fc

        # Make features
        response = self._get_response(params=params)

        matched = response.get('@iot.count')
        if matched:
            fc['numberMatched'] = matched

        # Query if values are less than expected
        features = self._fetch_to_limit(response, limit)

        crs_transform_func = get_transform_from_spec(crs_transform_spec)

        fc['features'] = [
            self._make_feature(
                feature=feature,
                select_properties=select_properties,
                skip_geometry=skip_geometry,
                crs_transform_func=crs_transform_func
            )
            for feature in features
        ]
        fc['numberReturned'] = len(features)

        return fc

    def _make_feature(
        self,
        feature,
        select_properties=[],
        skip_geometry=False,
        entity=None,
        crs_transform_func=None,
    ):
        """
        Private function: Create feature from entity

        :param feature: `dict` of STA entity
        :param select_properties: list of property names
        :param skip_geometry: bool of whether to skip geometry (default False)
        :param entity: SensorThings entity name
        :param crs_transform_func: `CrsTransform` function, optional

        :returns: dict of GeoJSON Feature
        """
        id = feature.get(self.id_field)
        f = {'type': 'Feature', 'id': id, 'properties': {}, 'geometry': None}

        # Fill properties block
        try:
            f['properties'] = self._expand_properties(
                feature, select_properties, entity
            )
        except KeyError as err:
            LOGGER.error(err)
            raise ProviderQueryError(err)

        # Make geometry
        if not skip_geometry:
            try:
                f['geometry'] = self._geometry(feature, entity)
            except (KeyError, IndexError):
                f['geometry'] = None

            if crs_transform_func is not None:
                crs_transform_feature(f, crs_transform_func)

        return f

    def _get_response(self, url=None, params={}, entity=None, expand=None):
        """
        Private function: Get STA response

        :param url: request url
        :param params: query parameters
        :param entity: SensorThings entity name
        :param expand: SensorThings expand query

        :returns: STA response
        """
        entity_ = entity or self.entity

        if url is None:
            url = url_join(self.data, entity_)

        params.update({'$expand': expand or self.expand[entity_]})

        r = self.http.get(url, params=params)
        LOGGER.debug(f'Request URL: {r.url}')

        if not r.ok:
            LOGGER.error(f'Bad http response code: {r.url}')
            raise ProviderConnectionError('Bad http response code')

        try:
            response = r.json()
        except JSONDecodeError as err:
            LOGGER.error('JSON decode error')
            raise ProviderQueryError(err)

        return response

    def _make_filter(self, properties, bbox=[], datetime_=None, entity=None):
        """
        Private function: Make STA filter from query properties

        :param properties: list of tuples (name, value)
        :param bbox: bounding box [minx,miny,maxx,maxy]
        :param datetime_: temporal (datestamp or extent)
        :param entity: SensorThings entity name

        :returns: STA $filter string of properties
        """

        filters = []
        for name, value in properties:
            filters.append(
                f'{name}/@iot.id eq {value}'
                if name in ENTITY
                else f'{name} eq {value}'
            )

        if bbox:
            filters.append(
                self._make_bbox(bbox, entity or self.entity)
            )

        if datetime_ is not None:
            filters.extend(
                self._parse_datetime(datetime_, self.time_field)
            )

        return ' and '.join(filters)

    @staticmethod
    def _parse_datetime(datetime_: str, time_field: str = None) -> list:
        """
        Private function: Parse datetime into STA filter

        :param datetime_: temporal (datestamp or extent)
        :param time_field: STA time field

        """

        if time_field is None:
            msg = 'time_field not enabled for collection'
            LOGGER.error(msg)
            raise ProviderQueryError(msg)

        filters = []

        if '/' in datetime_:
            time_start, time_end = datetime_.split('/')
            if time_start != '..':
                filters.append(f'{time_field} ge {fmtd(time_start)}')

            if time_end != '..':
                filters.append(f'{time_field} le {fmtd(time_end)}')
        else:

            filters.append(f'{time_field} eq {fmtd(datetime_)}')

        return filters

    def _make_bbox(self, bbox, entity):
        """
        Private function: Make STA bbox filter

        :param bbox: bounding box [minx,miny,maxx,maxy]
        :param entity: SensorThings entity name

        :returns: STA $filter string of bbox
        """
        entity = entity or self.entity

        minx, miny, maxx, maxy = bbox
        bbox_ = f'POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}))'  # noqa
        match entity:
            case 'Things':
                loc = 'Locations/location'
            case 'Datastreams':
                loc = 'Thing/Locations/location'
            case 'Observations':
                loc = 'FeatureOfInterest/feature'
            case 'ObservedProperties':
                loc = 'Datastreams/observedArea'
            case _:
                loc = 'location'

        return f"st_within({loc},geography'{bbox_}')"

    def _make_orderby(self, sortby):
        """
        Private function: Make STA filter from query properties

        :param sortby: list of dicts (property, order)

        :returns: STA $orderby string
        """

        orderby = []
        _map = {'+': 'asc', '-': 'desc'}
        for _ in sortby:
            prop = _['property']
            order = _map[_['order']]

            orderby.append(
                f'{prop}/@iot.id {order}'
                if prop in self.entity
                else f'{prop} {order}'
            )

        return ','.join(orderby)

    @staticmethod
    def to_iotid(identifier):
        """
        Private function: Safely format @iot.id value

        :param value: `str` of @iot.id value

        :returns: `str` of formatted @iot.id value
        """

        if isinstance(identifier, str):
            identifier2 = identifier.strip("'")
            return f"'{identifier2}'"

        else:
            return str(identifier)

    def _get_nextlink(self, response: dict) -> str:
        """
        Private function: Parse nextLink URL

        :param response: STA response with @iot.nextLink

        :returns: `str` of nextLink URL
        """
        next_ = (
            urlparse(response['@iot.nextLink'])
            ._replace(
                scheme=self.parsed_url.scheme,
                netloc=self.parsed_url.netloc
            )
            .geturl()
        )
        LOGGER.debug(f'Fetching next set of values: {next_}')
        return next_

    def _fetch_to_limit(self, response, limit: int | None = None):
        """
        Private function: Fetch nextLink results till limit

        :param response: STA response with @iot.nextLink
        :param limit: `int` of maximum number of results to fetch

        :returns: `list` of STA entities
        """

        values = response.get('value', [])
        # No limit, fetch all
        nolimit = limit is None

        while (limit and len(values) < limit) or nolimit:
            try:
                next_ = self._get_nextlink(response)
                response = self.http.get(next_).json()
                values.extend(response['value'])
            except (ProviderConnectionError, KeyError):
                break

        if limit and len(values) > limit:
            values = values[:limit]

        return values

    def _geometry(self, feature, entity=None):
        """
        Private function: Retrieve STA geometry

        :param feature: SensorThings entity
        :param entity: SensorThings entity name

        :returns: GeoJSON Geometry for Feature
        """

        entity = entity or self.entity

        match entity:
            case 'Things':
                locations = feature.get('Locations')
                location = locations[0]['location']

            case 'Observations':
                foi = feature['FeatureOfInterest']
                location = foi.get('feature')

            case 'Datastreams':
                try:
                    locations = feature['Thing'].get('Locations')
                    location = locations[0]['location']

                except (KeyError, IndexError):
                    foi = feature['Observations'][0]['FeatureOfInterest']
                    location = foi.get('feature')

            case 'ObservedProperties':
                locations = feature['Datastreams'][0]['Thing'].get('Locations')
                location = locations[0]['location']

            case _:
                LOGGER.warning('No geometry found')
                return None

        # SensorThings Geometry may be Feature GeoJSON or Geometry GeoJSON
        return (
            location['geometry'] if location['type'] == 'Feature' else location
        )

    def _expand_properties(self, feature, keys=(), uri='', entity=None):
        """
        Private function: Parse STA entity into feature

        :param feature: `dict` of SensorThings entity
        :param keys: keys used in properties block
        :param uri: uri of STA entity
        :param entity: SensorThings entity name

        :returns: dict of SensorThings feature properties
        """
        # Properties filter & display
        keys = (
            ()
            if not self.properties and not keys
            else set(self.properties) | set(keys)
        )

        if entity == 'Things':
            self._expand_location(feature)
        elif 'Thing' in feature.keys():
            self._expand_location(feature['Thing'])

        # Retain URI if present
        try:
            if self.uri_field is not None:
                uri = feature['properties'][self.uri_field]
        except KeyError:
            msg = f'Unable to find uri field: {self.uri_field}'
            LOGGER.error(msg)
            raise ProviderInvalidDataError(msg)

        # Create intra links
        for k, v in feature.items():
            if k in self.linked_entity:
                feature[k] = [
                    self._get_uri(_v, **self.linked_entity[k]) for _v in v
                ]
                LOGGER.debug(f'Created link for {k}')
            elif f'{k}s' in self.linked_entity:
                feature[k] = self._get_uri(v, **self.linked_entity[f'{k}s'])
                LOGGER.debug(f'Created link for {k}')

        # Make properties block
        if feature.get('properties'):
            feature.update(feature.pop('properties'))

        if keys:
            ret = {k: feature.pop(k) for k in keys}
            feature = ret

        if self.uri_field is not None:
            feature[self.uri_field] = uri

        return feature

    @staticmethod
    def _expand_location(entity):
        """
        Private function: Get STA item uri

        :param entity: `dict` of STA entity

        :returns: None
        """
        try:
            extra_props = entity['Locations'][0]['properties']
            entity['properties'].update(extra_props)
        except (KeyError, IndexError):
            selfLink = entity['@iot.selfLink']
            LOGGER.debug(f'{selfLink} has no Location properties')

    def _get_uri(self, entity, cnm, cid='@iot.id', uri=''):
        """
        Private function: Get STA item uri

        :param entity: `dict` of STA entity
        :param cnm: `str` of OAPI collection name
        :param cid: `str` of OAPI collection id field
        :param uri: `str` of STA entity uri field

        :returns: `str` of item uri
        """
        if uri:
            return entity['properties'][uri]
        else:
            id_ = self.to_iotid(entity[cid])
            uri = (self.rel_link, 'collections', cnm, 'items', id_)
            return url_join(*uri)

    @staticmethod
    def _get_entity(uri):
        """
        Private function: Parse STA Entity from uri

        :param uri: `str` of STA entity uri

        :returns: `str` of STA Entity
        """
        e = uri.split('/').pop()
        if e in ENTITY:
            return e
        else:
            return ''

    def _generate_mappings(self, provider_def: dict):
        """
        Generate mappings for the STA entity and set up intra-links.

        This function sets up the necessary mappings and configurations for
        the STA entity based on the provided provider definition.

        :param provider_def: `dict` of provider definition containing
                            configuration details for the STA entity.
        """
        self.data.rstrip('/')
        try:
            self.entity = provider_def['entity']
            self._url = url_join(self.data, self.entity)
        except KeyError:
            LOGGER.debug('Attempting to parse Entity from provider data')
            if not self._get_entity(self.data):
                raise RuntimeError('Entity type required')
            self.entity = self._get_entity(self.data)
            self._url = self.data
            self.data = self._url.rstrip(f'/{self.entity}')

        self.parsed_url = urlparse(self.data)

        # Default id
        if self.id_field:
            LOGGER.debug(f'Using id field: {self.id_field}')
        else:
            LOGGER.debug('Using default @iot.id for id field')
            self.id_field = '@iot.id'

        # Custom expand
        if provider_def.get('expand'):
            self.expand[self.entity] = provider_def['expand']

        # Create intra-links
        self.intralink = provider_def.get('intralink', False)
        if self.intralink and provider_def.get('rel_link'):
            # For pytest
            self.rel_link = provider_def['rel_link']

        elif not self.intralink:
            return

        # Read from pygeoapi config
        CONFIG = get_config()
        self.rel_link = get_base_url(CONFIG)

        for name, rs in CONFIG['resources'].items():
            pvs = rs.get('providers')
            if not pvs:
                continue

            p = get_provider_default(pvs)
            e = p.get('entity') or self._get_entity(p['data'])
            if any(
                [
                    not pvs,  # No providers in resource
                    not p.get('intralink'),  # No configuration for intralinks
                    not e,  # No STA entity found
                    self.data not in p.get('data'),  # No common STA endpoint
                ]
            ):
                continue

            if p.get('uri_field'):
                LOGGER.debug(f'Linking {e} with field: {p["uri_field"]}')
            else:
                LOGGER.debug(f'Linking {e} with collection: {name}')

            self.linked_entity[e] = {
                'cnm': name,  # OAPI collection name,
                'cid': p.get('id_field', '@iot.id'),  # OAPI id_field
                'uri': p.get('uri_field'),  # STA uri_field
            }

    def __repr__(self):
        return f'<SensorThingsProvider> {self.data}, {self.entity}'
