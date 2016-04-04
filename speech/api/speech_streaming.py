#!/usr/bin/python

import contextlib
import threading

from gcloud.credentials import get_credentials

from google.cloud.speech.v1.cloud_speech_pb2 import *
from google.rpc import code_pb2

from grpc.beta import implementations

import pyaudio

# Audio recording parameters
RATE = 16000
CHANNELS = 1
CHUNK = 2048

# Keep the request alive for this many seconds
DEADLINE_SECS = 8 * 60 * 60
SPEECH_SCOPE = 'https://www.googleapis.com/auth/xapi.zoo'


def _make_channel(host, port):
    """Creates an SSL channel with auth credentials from the environment."""
    # In order to make an https call, use an ssl channel with defaults
    ssl_channel = implementations.ssl_channel_credentials(None, None, None)

    # Grab application default credentials from the environment
    creds = get_credentials().create_scoped([SPEECH_SCOPE])
    # Add a plugin to inject the creds into the header
    auth_header = (
            'Authorization',
            'Bearer ' + creds.get_access_token().access_token)
    auth_plugin = implementations.metadata_call_credentials(
            lambda _, cb: cb([auth_header], None),
            name='google_creds')

    # compose the two together for both ssl and google auth
    composite_channel = implementations.composite_channel_credentials(
            ssl_channel, auth_plugin)

    return implementations.secure_channel(host, port, composite_channel)


@contextlib.contextmanager
def _record_audio(channels, rate, chunk):
    """Opens a recording stream in a context manager."""
    p = pyaudio.PyAudio()
    audio_stream = p.open(
        format=pyaudio.paInt16, channels=channels, rate=rate,
        input=True, frames_per_buffer=chunk,
    )

    yield audio_stream

    audio_stream.stop_stream()
    audio_stream.close()
    p.terminate()


def _request_stream(stop_audio, channels=CHANNELS, rate=RATE, chunk=CHUNK):
    """Yields `RecognizeRequest`s constructed from a recording audio stream.

    Args:
        stop_audio: A threading.Event object stops the recording when set.
        channels: How many audio channels to record.
        rate: The sampling rate.
        chunk: Buffer audio into chunks of this size before sending to the api.
    """
    with _record_audio(channels, rate, chunk) as audio_stream:
        # The initial request must contain metadata about the stream, so the
        # server knows how to interpret it.
        metadata = InitialRecognizeRequest(
            encoding='LINEAR16', sample_rate=rate)
        audio_request = AudioRequest(content=audio_stream.read(chunk))

        yield RecognizeRequest(
            initial_request=metadata,
            audio_request=audio_request)

        while not stop_audio.is_set():
            # Subsequent requests can all just have the content
            audio_request = AudioRequest(content=audio_stream.read(chunk))

            yield RecognizeRequest(audio_request=audio_request)

    raise StopIteration()


def main():
    stop_audio = threading.Event()
    with beta_create_Speech_stub(
            _make_channel('speech.googleapis.com', 443)) as service:
        try:
            for resp in service.Recognize(
                    _request_stream(stop_audio), DEADLINE_SECS):
                if resp.error.code != code_pb2.OK:
                    raise Exception('Server error: ' + resp.error.message)

                # Display the transcriptions & their alternatives
                for result in resp.results:
                    print(result.alternatives)

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if any(alt.confidence > .5 and
                       (alt.transcript.strip() in ('exit', 'quit'))
                       for result in resp.results
                       for alt in result.alternatives):
                    print('Exiting..')
                    stop_audio.set()
        finally:
            # On exceptions (such as timeout), stop the audio stream.
            stop_audio.set()


if __name__ == '__main__':
    main()
