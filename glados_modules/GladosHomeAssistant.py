
# 3rd Party Imports
from homeassistant_api import Client

# glados imports
from glados_modules.GlogConfig import setup_logger


class HomeAssistantLink:
    def __init__(self, config_file):
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__)
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

    # TODO need to figure out how were going to sync the camera "scan / hunt" function for new people


""" stub code for camera body hunt
 def target_scan(self, target="person", search_time=90, confidence=.70):
     self.logger.debug(f"Camera Scanning for target: {target}")
     target_found = False
     t = time()
     while (time() - t) < search_time and target_found is False:
         if target in self.results.keys():
             for p in self.results[target]['objects']:
                 if p['confidence'] >= confidence:
                     # found the target in the timeframe
                     target_found = True
                     self.logger.debug(f"Camera Found target: {target} , {p}")
                     break
"""

