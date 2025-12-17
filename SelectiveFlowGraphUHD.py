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
from PyQt5.QtCore import QObject, pyqtSlot
from gnuradio import analog
from gnuradio import audio
from gnuradio import blocks, gr
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
from gnuradio import uhd
import time
import sip
import threading



class SelectiveFlowGraphUHD(gr.top_block, Qt.QWidget):

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

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "SelectiveFlowGraphUHD")

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
        self.squelch = squelch = -50
        self.samp_rate_0 = samp_rate_0 = 2e6
        self.samp_rate = samp_rate = 32000
        self.quadrature = quadrature = 500000
        self.freq_ch = freq_ch = 161812500
        self.freq = freq = 161812500
        self.cutoff = cutoff = 10e3
        self.band = band = int(25e2)
        self.audio_samp_rate = audio_samp_rate = 48000

        ##################################################
        # Blocks
        ##################################################

        self._squelch_range = qtgui.Range(-150, 20, 1, -50, 200)
        self._squelch_win = qtgui.RangeWidget(self._squelch_range, self.set_squelch, "squelch", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._squelch_win)
        # Create the options list
        self._freq_ch_options = [161812500, 161862500, 446043750, 99300000]
        # Create the labels list
        self._freq_ch_labels = ['URG TN', '118 ELI', 'PRM CH4 SEL', 'FM RADIO']
        # Create the combo box
        self._freq_ch_tool_bar = Qt.QToolBar(self)
        self._freq_ch_tool_bar.addWidget(Qt.QLabel("Frequency" + ": "))
        self._freq_ch_combo_box = Qt.QComboBox()
        self._freq_ch_tool_bar.addWidget(self._freq_ch_combo_box)
        for _label in self._freq_ch_labels: self._freq_ch_combo_box.addItem(_label)
        self._freq_ch_callback = lambda i: Qt.QMetaObject.invokeMethod(self._freq_ch_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._freq_ch_options.index(i)))
        self._freq_ch_callback(self.freq_ch)
        self._freq_ch_combo_box.currentIndexChanged.connect(
            lambda i: self.set_freq_ch(self._freq_ch_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._freq_ch_tool_bar)
        # Create the options list
        self._band_options = [2500, 500, 38000]
        # Create the labels list
        self._band_labels = ['NB PMR 12.5k', 'NB VHF 12k', 'WB FM']
        # Create the combo box
        self._band_tool_bar = Qt.QToolBar(self)
        self._band_tool_bar.addWidget(Qt.QLabel("Band" + ": "))
        self._band_combo_box = Qt.QComboBox()
        self._band_tool_bar.addWidget(self._band_combo_box)
        for _label in self._band_labels: self._band_combo_box.addItem(_label)
        self._band_callback = lambda i: Qt.QMetaObject.invokeMethod(self._band_combo_box, "setCurrentIndex", Qt.Q_ARG("int", self._band_options.index(i)))
        self._band_callback(self.band)
        self._band_combo_box.currentIndexChanged.connect(
            lambda i: self.set_band(self._band_options[i]))
        # Create the radio buttons
        self.top_layout.addWidget(self._band_tool_bar)
        self._volume_range = qtgui.Range(0.1, 2, 0.1, 1, 200)
        self._volume_win = qtgui.RangeWidget(self._volume_range, self.set_volume, "Volume", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._volume_win)
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))

        self.uhd_usrp_source_0.set_center_freq(freq, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_gain(0, 0)
        self.selcal_selcall_decoder_0 = selcal.selcal_decoder(
            sample_rate=audio_samp_rate,
            protocol='CCIR-1',
            target_code='11501',
            code_length=5,
            tone_duration_ms=0.0,
            debug=False
        )
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
        self.qtgui_waterfall_sink_x_0 = qtgui.waterfall_sink_c(
            1024, #size
            window.WIN_HAMMING, #wintype
            freq_ch, #fc
            audio_samp_rate, #bw
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_waterfall_sink_x_0.set_update_time(0.10)
        self.qtgui_waterfall_sink_x_0.enable_grid(False)
        self.qtgui_waterfall_sink_x_0.enable_axis_labels(True)



        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        colors = [0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_waterfall_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_waterfall_sink_x_0.set_color_map(i, colors[i])
            self.qtgui_waterfall_sink_x_0.set_line_alpha(i, alphas[i])

        self.qtgui_waterfall_sink_x_0.set_intensity_range(-140, 10)

        self._qtgui_waterfall_sink_x_0_win = sip.wrapinstance(self.qtgui_waterfall_sink_x_0.qwidget(), Qt.QWidget)

        self.top_layout.addWidget(self._qtgui_waterfall_sink_x_0_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_HAMMING, #wintype
            freq_ch, #fc
            audio_samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(True)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(True)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(
                1,
                samp_rate,
                cutoff,
                transition,
                window.WIN_HAMMING,
                6.76))
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
        self.audio_sink_0_0 = audio.sink(audio_samp_rate, '', True)
        self.audio_sink_0 = audio.sink(audio_samp_rate, '', True)
        self.analog_pwr_squelch_xx_0 = analog.pwr_squelch_cc(squelch, (1e-4), 0, True)
        self.analog_nbfm_rx_0 = analog.nbfm_rx(
        	audio_rate=50000,
        	quad_rate=quadrature,
        	tau=(75e-6),
        	max_dev=band,
          )


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.selcal_selcall_decoder_0, 'selcall_out'), (self.blocks_message_debug_0, 'print'))
        self.connect((self.analog_nbfm_rx_0, 0), (self.rational_resampler_xxx_0_0_0, 0))
        self.connect((self.analog_pwr_squelch_xx_0, 0), (self.analog_nbfm_rx_0, 0))
        self.connect((self.band_pass_filter_0, 0), (self.selcal_selcall_decoder_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.analog_pwr_squelch_xx_0, 0))
        self.connect((self.rational_resampler_xxx_0_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0, 0), (self.audio_sink_0_0, 0))
        self.connect((self.rational_resampler_xxx_0_0_0, 0), (self.band_pass_filter_0, 0))
        self.connect((self.selcal_selcall_decoder_0, 0), (self.audio_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.qtgui_waterfall_sink_x_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.rational_resampler_xxx_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "SelectiveFlowGraphUHD")
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

    def get_samp_rate_0(self):
        return self.samp_rate_0

    def set_samp_rate_0(self, samp_rate_0):
        self.samp_rate_0 = samp_rate_0

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, self.cutoff, self.transition, window.WIN_HAMMING, 6.76))

    def get_quadrature(self):
        return self.quadrature

    def set_quadrature(self, quadrature):
        self.quadrature = quadrature

    def get_freq_ch(self):
        return self.freq_ch

    def set_freq_ch(self, freq_ch):
        self.freq_ch = freq_ch
        self._freq_ch_callback(self.freq_ch)
        self.qtgui_waterfall_sink_x_0.set_frequency_range(self.freq_ch, self.audio_samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.freq_ch, self.audio_samp_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)

    def get_cutoff(self):
        return self.cutoff

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, self.cutoff, self.transition, window.WIN_HAMMING, 6.76))

    def get_band(self):
        return self.band

    def set_band(self, band):
        self.band = band
        self.analog_nbfm_rx_0.set_max_deviation(self.band)
        self._band_callback(self.band)

    def get_audio_samp_rate(self):
        return self.audio_samp_rate

    def set_audio_samp_rate(self, audio_samp_rate):
        self.audio_samp_rate = audio_samp_rate
        self.qtgui_waterfall_sink_x_0.set_frequency_range(self.freq_ch, self.audio_samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.freq_ch, self.audio_samp_rate)
        self.band_pass_filter_0.set_taps(firdes.band_pass(1, self.audio_samp_rate, 300, 38e2, 200, window.WIN_HAMMING, 6.76))




def main(top_block_cls=SelectiveFlowGraphUHD, options=None):

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
