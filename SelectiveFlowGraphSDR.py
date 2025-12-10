#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: pietro
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import audio
from gnuradio import blocks, gr
from gnuradio import bomboklat
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import selcal
import osmosdr
import time
import threading



class SelectiveFlowGraphSDR(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "SelectiveFlowGraphSDR")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.volume = volume = 1
        self.transition = transition = 1e6
        self.squelch = squelch = -20
        self.samp_rate = samp_rate = 2e6
        self.quadrature = quadrature = 500000
        self.freq = freq = 446043750
        self.cutoff = cutoff = 10e3
        self.audio_samp_rate = audio_samp_rate = 48000

        ##################################################
        # Blocks
        ##################################################

        self._volume_range = qtgui.Range(0.1, 2, 0.1, 1, 200)
        self._volume_win = qtgui.RangeWidget(self._volume_range, self.set_volume, "Volume", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._volume_win)
        self._squelch_range = qtgui.Range(-150, 20, 1, -20, 200)
        self._squelch_win = qtgui.RangeWidget(self._squelch_range, self.set_squelch, "squelch", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._squelch_win)
        self.selcal_selcall_decoder_0 = selcal.selcal_decoder(
            sample_rate=audio_samp_rate,
            protocol='ZVEI-2',
            target_code='18515',
            code_length=5,
            tone_duration_ms=0.0,
            debug=False
        )
        self.rtlsdr_source_0 = osmosdr.source(
            args="numchan=" + str(1) + " " + ""
        )
        self.rtlsdr_source_0.set_time_unknown_pps(osmosdr.time_spec_t())
        self.rtlsdr_source_0.set_sample_rate(samp_rate)
        self.rtlsdr_source_0.set_center_freq(freq, 0)
        self.rtlsdr_source_0.set_freq_corr(0, 0)
        self.rtlsdr_source_0.set_dc_offset_mode(0, 0)
        self.rtlsdr_source_0.set_iq_balance_mode(0, 0)
        self.rtlsdr_source_0.set_gain_mode(False, 0)
        self.rtlsdr_source_0.set_gain(10, 0)
        self.rtlsdr_source_0.set_if_gain(20, 0)
        self.rtlsdr_source_0.set_bb_gain(20, 0)
        self.rtlsdr_source_0.set_antenna('', 0)
        self.rtlsdr_source_0.set_bandwidth(0, 0)
        self.rational_resampler_xxx_0_0_0 = filter.rational_resampler_fff(
                interpolation=24,
                decimation=25,
                taps=[],
                fractional_bw=0)
        self.rational_resampler_xxx_0_0 = filter.rational_resampler_ccc(
                interpolation=1,
                decimation=4,
                taps=[],
                fractional_bw=0)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                cutoff,
                transition,
                window.WIN_HAMMING,
                6.76))
        self.bomboklat_mimibombo_0 = bomboklat.mimibombo(gain=volume)
        self.blocks_message_debug_0 = blocks.message_debug(True, gr.log_levels.info)
        self.band_pass_filter_0 = filter.fir_filter_fff(
            1,
            firdes.band_pass(
                1,
                audio_samp_rate,
                300,
                38e2,
                200,
                window.WIN_HAMMING,
                6.76))
        self.audio_sink_0 = audio.sink(audio_samp_rate, '', True)
        self.analog_pwr_squelch_xx_0 = analog.pwr_squelch_cc(squelch, (1e-4), 0, True)
        self.analog_nbfm_rx_0 = analog.nbfm_rx(
        	audio_rate=50000,
        	quad_rate=quadrature,
        	tau=(75e-6),
        	max_dev=5e3,
          )


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.selcal_selcall_decoder_0, 'selcall_out'), (self.blocks_message_debug_0, 'print'))
        self.connect((self.analog_nbfm_rx_0, 0), (self.rational_resampler_xxx_0_0_0, 0))
        self.connect((self.analog_pwr_squelch_xx_0, 0), (self.analog_nbfm_rx_0, 0))
        self.connect((self.band_pass_filter_0, 0), (self.selcal_selcall_decoder_0, 0))
        self.connect((self.bomboklat_mimibombo_0, 0), (self.audio_sink_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.analog_pwr_squelch_xx_0, 0))
        self.connect((self.rational_resampler_xxx_0_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0, 0), (self.band_pass_filter_0, 0))
        self.connect((self.rtlsdr_source_0, 0), (self.rational_resampler_xxx_0_0, 0))
        self.connect((self.selcal_selcall_decoder_0, 0), (self.bomboklat_mimibombo_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "SelectiveFlowGraphSDR")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_volume(self):
        return self.volume

    def set_volume(self, volume):
        self.volume = volume

    def get_transition(self):
        return self.transition

    def set_transition(self, transition):
        self.transition = transition
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, self.cutoff, self.transition, window.WIN_HAMMING, 6.76))

    def get_squelch(self):
        return self.squelch

    def set_squelch(self, squelch):
        self.squelch = squelch
        self.analog_pwr_squelch_xx_0.set_threshold(self.squelch)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, self.cutoff, self.transition, window.WIN_HAMMING, 6.76))
        self.rtlsdr_source_0.set_sample_rate(self.samp_rate)

    def get_quadrature(self):
        return self.quadrature

    def set_quadrature(self, quadrature):
        self.quadrature = quadrature

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.rtlsdr_source_0.set_center_freq(self.freq, 0)

    def get_cutoff(self):
        return self.cutoff

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, self.cutoff, self.transition, window.WIN_HAMMING, 6.76))

    def get_audio_samp_rate(self):
        return self.audio_samp_rate

    def set_audio_samp_rate(self, audio_samp_rate):
        self.audio_samp_rate = audio_samp_rate
        self.band_pass_filter_0.set_taps(firdes.band_pass(1, self.audio_samp_rate, 300, 38e2, 200, window.WIN_HAMMING, 6.76))




def main(top_block_cls=SelectiveFlowGraphSDR, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
