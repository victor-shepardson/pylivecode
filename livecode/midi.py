"MIDI input functionality"

import logging


from . pattern import Vars
M = Vars()

try:
    import mido
except ImportError:
    logging.warning('MIDI unavailable; install mido and python-rtmidi')

try:
    def midi_callback(m):
        logging.debug(m)
        if m.channel != midi_channel:
            return
        if m.type=='control_change':
            var = 'cc'+str(m.control)
            M[var].set(m.value)
            logging.debug((var, m.value))

    def midi(*ports, channel=0):
        global midi_channel, midi_port
        midi_channel = channel
        midi_port = mido.ports.MultiPort([
                mido.open_input(port, callback=midi_callback)
                for port in ports or mido.get_input_names()
            ])
    midi()
except Exception as e:
    logging.warning(e)
