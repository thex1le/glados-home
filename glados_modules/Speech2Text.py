from threading import Thread
from time import sleep
import multiprocessing as mp

#3rd party
import regex as re
import speech_recognition as sr

# glados imports
from glados_modules.GlogConfig import setup_logger


class GladosSTT(Thread):
    # glados speach to text
    def __init__(self, glocal) -> None:
        Thread.__init__(self)
        Thread.daemon = True
        self.__name__ = self.__class__.__name__
        self.logger = setup_logger(name=self.__name__)
        self.text = None
        self.glocal = glocal
        self.mplist = list()

    def get_text(self):
        # return the text and sent it back to none for next question
        try:
            text = self.mplist.pop()
        except IndexError:
            text = None
        return text

    def parse_command(self, user_prompt: str) -> dict:
        glados_pattern = r'(hey glados){e<=3}'
        glados_match = re.search(glados_pattern, user_prompt, re.IGNORECASE | re.BESTMATCH)
        if glados_match:
            split_index = glados_match.end()
            greeting = user_prompt[:split_index].strip()
            command = user_prompt[split_index:].strip()
            has_extra_command = bool(command)
            return {"greeting": greeting, "has_extra_command": has_extra_command, "command": command}
        else:
            return {"greeting": None, "has_extra_command": False, "command": None}

    def record(self, mp_list):
        # TODO, how do we keep things local so were not hitting google all the time...
        while True:
            msg = "Say 'Hey GLaDOS' to start recording your question"
            self.logger.info(msg)
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                self.logger.debug("Adjusting for noise")
                recognizer.adjust_for_ambient_noise(source, .5)
                self.logger.debug("getting audio")
                audio = recognizer.listen(source)
                self.logger.debug("audio done")
                try:
                    transcription = recognizer.recognize_google(audio)
                    self.logger.debug(f"transcribe done, {transcription.lower()}")
                    pcommand = self.parse_command(transcription)
                    self.logger.debug(f'parse command is {pcommand}')
                    if pcommand["greeting"] is not None:
                        # here is where we should pause and take a longer recording
                        # for the command we need to trigger glados to talk here...
                        # TODO reconsider how this works with multithreading
                        self.logger.debug(pcommand)
                        if pcommand["has_extra_command"] is False:
                            greet = self.glocal.random_greeting(True)
                            rq = self.glocal.random_question(True)
                            self.glocal.speak(f"{greet}. {rq}")
                            with sr.Microphone() as source:
                                recognizer = sr.Recognizer()
                                source.pause_threshold = 1
                                audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                                transcription = recognizer.recognize_google(audio)
                            self.logger.debug("good user_prompt")
                            # transcript audio to test
                            # check for cancel command
                            # TODO work out how the cancel command works
                            if self.glocal._gladosLocal__check_local_command(transcription.lower(),
                                                                             re.compile(r'cancel?')) is True:
                                self.logger.debug('cancel true')
                                self.glocal.random_cancel_response()
                                continue
                        self.logger.debug(transcription)
                        mp_list.append(pcommand['command'])
                except Exception as e:
                    self.logger.error(f"An unknown error occurred: {e}")

    def run(self):
        # use manager to run management loop
        with mp.Manager() as manager:
            self.mplist = manager.list()
            self.proc = mp.Process(target=self.record, args=(self.mplist,))
            self.proc.start()
            while True:
                sleep(10)

