
# 3rd Party Imports
from homeassistant_api import Client

# glados imports
from glados_modules.GlogConfig import setup_logger
from glados_modules.GladosEnums import LoggingEnums


class HomeAssistantLink:
    def __init__(self, config_file):
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__, console_logging=LoggingEnums.LOG_LEVEL_INFO.value)
        base = config_file['HOMEASSISTANT']
        self.token = base['token']
        self.api = base['api']
        self.weather_entity_id = base['weather_entity']

    def __get_weather(self) -> (None, dict):
        client = Client(self.api, self.token)
        data = None
        try:
            # Fetch the state of the weather entity
            weather_data = client.get_entity(entity_id=self.weather_entity_id)
            if weather_data:
                data = weather_data
        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
        return data

    def get_temp(self) -> str:
        """
        Return current temp highs and low's as a string
        """
        wdata = self.__get_weather()
        watt = wdata.state.attributes
        return f"The current temperature is {watt['temperature']}"

