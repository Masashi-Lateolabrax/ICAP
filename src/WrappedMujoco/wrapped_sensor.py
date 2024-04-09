class WrappedSensor:
    def __init__(self, mj_data_sensor_views):
        self._raw = mj_data_sensor_views

    def get_data(self):
        return self._raw.data

    def get_id(self):
        return self._raw.id

    def get_name(self):
        return self._raw.name
