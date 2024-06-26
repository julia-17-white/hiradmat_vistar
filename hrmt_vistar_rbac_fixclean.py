# -*- coding: utf-8 -*-
import pyjapc
from scipy.ndimage import median_filter as img_filter
from scipy.optimize import curve_fit, OptimizeWarning
from functools import partial, wraps
import numpy as np
from datetime import datetime
from PyQt5.QtTest import QTest
from PyQt5.QtCore import pyqtSignal, QTimer, QObject, QThread
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
from scipy.stats import multivariate_normal
import random
import os
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import signal
import warnings
import pyrbac
import pylogbook
import argparse

from typing import List, Union

signal.signal(signal.SIGINT, signal.SIG_DFL)

warnings.filterwarnings("ignore", message='No Python match found for JAPC.*')
matplotlib.use('Agg')


def get_fit(x, y, maxfev=100) -> tuple:
    guess = [
        np.max(y),
        x[np.argmax(y)],
        0.5,
        np.mean(y[0:10])
    ]
    bounds = [
        [0, np.min(x), 0, 0],
        [np.inf, np.max(x), np.inf, np.inf]
    ]
    popt, pcov = curve_fit(gaus1d, x, y, p0=guess, bounds=bounds, maxfev=maxfev)
    return x, gaus1d(x, *popt), popt, pcov


def get_filter(filter_pos):
    filters = ['OD0', 'OD1', 'OD2', 'OD3', 'OD4']
    filter_wheel_positions = np.arange(0, 5) * 1200
    idx = np.argmin(np.abs(filter_wheel_positions - filter_pos))
    return filters[idx]


def get_screen(screen_pos):
    screens = ['Screen Out', 'Screen Out', 'CNT#stretched', 'CNT#high-purity', 'Glassy Carbon', 'Chromox Ref']
    screen_motor_positions = np.array([0, 13867, 27200, 40533, 53866, 67222])
    idx = np.argmin(np.abs(screen_motor_positions - screen_pos))
    return screens[idx]


def gaus1d(x, a, x0, sigma, c) -> np.ndarray:
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c


def unix_ts_to_dt(ts: int):
    return pd.to_datetime(ts, unit='us', utc=True).tz_convert('Europe/Brussels')


def print_unix_ts(ts: int) -> str:
    return f'{unix_ts_to_dt(ts):%y-%m-%d %H:%M:%S.%f}'


def standard_subscription(device, japc, updater: pyqtSignal):
    sh = japc.subscribeParam(
        f'{device.device_name}/{device.japc_field}',
        partial(device.callback, updater),
        timingSelectorOverride=device.selector,
        getHeader=True
    )
    return sh


# We use a method wrapper so selector, acquisition and cycle timestamp are set without the need to write this within every single callback.
# Truly my proudest moment of coding.
def primary_callback_wrap(method):
    @wraps(method)
    def wrapper(self, *method_args, **method_kwargs):
        if method_args[-1]['isFirstUpdate']:
            print(f'Successfully subscribed to {method_args[-3]}!')
            return

        self.selector = method_args[-1]['selector']
        self.acq_ts = int(method_args[-1]['acqStamp'].timestamp()*1e6)

        if issubclass(self.__class__, PpmDevice):
            self.cycle_ts = int(method_args[-1]['cycleStamp'].timestamp()*1e6)

        return method(self, *method_args, **method_kwargs)
    return wrapper


# This method wrapper is only used for secondary callbacks, e.g., filter and screen of the BTV (which is provided by add. FESA devices)
def secondary_callback_wrap(method):
    @wraps(method)
    def wrapper(self, *method_args, **method_kwargs):
        if method_args[-1]['isFirstUpdate']:
            print(f'Successfully subscribed to {method_args[-3]}!')
            return

        return method(self, *method_args, **method_kwargs)
    return wrapper


# For all of the following device classes we could theoretically use @dataclass decorator to get rid of __init__
# But due to the python version of acc-py we cannot mix non-default and default keyword arguments during inheritance:
# https://stackoverflow.com/a/53085935
# Would save a few tens of lines of code
class Device:
    """Class for a generic (non-ppm) device"""
    acq_ts: int = int(datetime.utcnow().timestamp()*1e6)
    selector = ''

    def __init__(self, name: str, device_name: str, japc_field: str):
        self.name = name
        self.device_name = device_name
        self.japc_field = japc_field

    def __str__(self):
        return self.name

    def create_subscriptions(self, japc, updater: pyqtSignal):
        return [standard_subscription(self, japc, updater)]


class PpmDevice(Device):
    """Class for a ppm device"""
    cycle_ts: int = int(datetime.utcnow().timestamp()*1e6)

    def __init__(self, name: str, device_name: str, japc_field: str, selector: str):
        super().__init__(name, device_name, japc_field)
        self.selector = selector


class BtvDigitalCamera(Device):
    """Class for BtvDc"""
    img: np.ndarray = np.zeros((2, 2))
    img_h: np.ndarray = np.zeros(2)
    img_v: np.ndarray = np.zeros(2)
    filter_pos: float = 0.0
    screen_pos: float = 0.0
    od_filter: str = get_filter(filter_pos)
    screen: str = get_screen(screen_pos)
    cycle_ts: int = 0

    def __init__(self, name: str, device_name: str, filter_channel: str, screen_channel: str, japc_field: str = 'LastImage'):
        super().__init__(name, device_name, japc_field)
        self.filter_channel = filter_channel
        self.screen_channel = screen_channel

    # def get_proj(self) -> tuple:
    #     return (self.img_h, np.sum(self.img, axis=0)), (self.img_v, np.sum(self.img, axis=1))

    def get_img(self) -> tuple:
        return self.img_h, self.img_v, self.img

    @primary_callback_wrap
    def callback(self, updater: pyqtSignal, param: str, value: dict, header: dict):
        self.cycle_ts = int(value['cycleStampGlob'] / 1000)
        self.selector = value['cycleNameGlob']

        self.img = value['image2D']
        self.img_h = value['imagePositionSet1']
        self.img_v = value['imagePositionSet2']

        updater.emit()

    @secondary_callback_wrap
    def callback_filter(self, param: str, value: dict, header: dict):
        self.filter_pos = value['position']
        self.od_filter = get_filter(value['position'])

    @secondary_callback_wrap
    def callback_screen(self, param: str, value: dict, header: dict):
        self.screen_pos = value['position']
        self.screen = get_screen(value['position'])

    def create_subscriptions(self, japc, updater: pyqtSignal):
        shs = [
            standard_subscription(self, japc, updater),
            japc.subscribeParam(
                f'{self.filter_channel}/Acquisition',
                self.callback_filter,
                timingSelectorOverride='',
                getHeader=True
            ),
            japc.subscribeParam(
                f'{self.screen_channel}/Acquisition',
                self.callback_screen,
                timingSelectorOverride='',
                getHeader=True
            )
        ]

        return shs


class BtvICamera(PpmDevice):
    """Class for BtvI"""
    img: np.ndarray = np.zeros((2, 2))
    img_h: np.ndarray = np.zeros(2)
    img_v: np.ndarray = np.zeros(2)
    od_filter: str = 'no_filter'
    screen: str = 'no_screen'

    def __init__(self, name: str, device_name: str, selector: str = 'SPS.USER.HIRADMT*', japc_field: str = 'Image'):
        super().__init__(name, device_name, japc_field, selector)

    def get_img(self) -> tuple:
        return self.img_h, self.img_v, self.img

    @primary_callback_wrap
    def callback(self, updater: pyqtSignal, param: str, value: dict, header: dict):
        if header['isFirstUpdate']:
            return

        self.img = value['imageSet']
        self.img_h = value['imagePositionSet1']
        self.img_v = value['imagePositionSet2']
        self.img = self.img.reshape((len(self.img_v), len(self.img_h)))

        self.od_filter = value['filterSelectStr'][value['filterSelect'][0]]
        self.screen = value['screenSelectStr'][value['screenSelect'][0]]

        updater.emit()


class BeamCurrentTransformer(PpmDevice):
    """Class for a Beam Current Transformer"""
    bunchIntensity = np.zeros(100)
    totalIntensity: float = 0
    ExtractionFlag: bool = False

    def __init__(self, name: str, device_name: str, selector: str = 'SPS.USER.HIRADMT*', japc_field: str = 'CaptureAcquisition'):
        super().__init__(name, device_name, japc_field, selector)

    @primary_callback_wrap
    def callback(self, updater: pyqtSignal, param: str, value: dict, header: dict):
        if type(value['totalIntensity']) == float:
            self.totalIntensity = value['totalIntensity'] * (10 ** value['totalIntensity_unitExponent'])
        else:
            self.totalIntensity = 0

        if len(value['bunchIntensity']) > 0:
            self.bunchIntensity = value['bunchIntensity'] * 10 ** value['bunchIntensity_unitExponent']
        else:
            self.bunchIntensity = 0

        self.ExtractionFlag = True if self.totalIntensity > 1e9 else False

        updater.emit()


class BeamPositionMonitor(PpmDevice):
    """Class for a Beam Position Monitor"""
    positions: dict = {}

    def __init__(self, name: str, device_name: str, bpms: List[str], selector: str = 'SPS.USER.HIRADMT*', japc_field: str = 'Acquisition'):
        super().__init__(name, device_name, japc_field, selector)
        self.bpms = bpms
        for bpm in self.bpms:
            self.positions[f'{bpm}.H'] = np.random.rand(30)*100
            self.positions[f'{bpm}.V'] = np.random.rand(30)*50

    @primary_callback_wrap
    def callback(self, updater: pyqtSignal, param: str, value: dict, header: dict):
        channels = value['channelNames'].tolist()
        channels_idx = [channels.index(value) for value in channels if any(bpm in value for bpm in self.bpms)]
        channel_values = value['bunchPositions'].tolist()
        for idx in channels_idx:
            self.positions[channels[idx]] = channel_values[idx]

        updater.emit()

    def get_bpms(self) -> List[str]:
        bpms = list(self.positions.keys())
        bpms = [bpm[:-2] for bpm in bpms]
        bpms = list(set(bpms))
        return bpms

    def get_positions(self, bpm) -> tuple:
        pos_h = self.positions[f'{bpm}.H']
        pos_v = self.positions[f'{bpm}.V']
        pos_h = pos_h[pos_h != 0]
        pos_v = pos_v[pos_v != 0]

        if (type(pos_h) != list) & (type(pos_v) != list):
            pos_h = [pos_h]
            pos_v = [pos_v]

        return pos_h, pos_v


class BeamQualityMonitor(PpmDevice):
    """Class for a Beam Quality Monitor"""
    bunchlengths: np.ndarray = np.zeros(1)
    bunchlength_mean: float = 0.0
    bunchlength_std: float = 0.0
    nbunches: int = 0

    def __init__(self, name: str, device_name: str, selector: str = 'SPS.USER.HIRADMT*', japc_field: str = 'Acquisition'):
        super().__init__(name, device_name, japc_field, selector)

    @primary_callback_wrap
    def callback(self, updater: pyqtSignal, param: str, value: dict, header: dict):
        bunchlengths = value['bunchLengths']

        if type(bunchlengths) == float:
            self.nbunches = 1
        else:
            self.nbunches = len(bunchlengths[bunchlengths > 0])
            self.bunchlengths = bunchlengths[bunchlengths > 0]

        self.bunchlengths = bunchlengths
        self.bunchlength_mean = np.mean(bunchlengths)
        self.bunchlength_std = np.std(bunchlengths)

        updater.emit()


class BeamLossMonitor(PpmDevice):
    normLosses = {}

    def __init__(self, name: str, device_name: str, selector: str = 'SPS.USER.HIRADMT*', japc_field: str = 'Acquisition'):
        super().__init__(name, device_name, japc_field, selector)

    @primary_callback_wrap
    def callback(self, updater: pyqtSignal, param: str, value: dict, header: dict):
        # some strange pyjapc conventions
        if len(value) < 5:
            value = value[0]

        blms = value['channelNames'].tolist()
        blm_idx = [blms.index(value) for value in blms if 'TT66' in value]
        losses = [value['normLosses'][idx] for idx in blm_idx]
        TT66_BLMs = [blms[idx] for idx in blm_idx]

        BLM_losses = [loss for _, loss in sorted(zip(TT66_BLMs, losses), key=lambda pair: pair[0].split('.')[2])]
        BLMs = [BLM for BLM in sorted(TT66_BLMs, key=lambda x: x.split('.')[2])]

        for blm, value in zip(BLMs, BLM_losses):
            self.normLosses[blm[3:-8]] = value

        # Call update function/trigger timer
        updater.emit()

    def plot_BLMs(self):
        blm_ts = print_unix_ts(self.acq_ts)
        cycle_ts = print_unix_ts(self.cycle_ts)

        fig, ax = plt.subplots(figsize=(6.8, 6.8 / 4 * 3), dpi=200)

        blm_names = list(self.normLosses.keys())
        normLosses = list(self.normLosses.values())

        ax.plot(blm_names, normLosses, 's-')
        ax.set_xticklabels(blm_names, rotation=45, ha='right')


        max_loss = np.max(normLosses) if np.max(normLosses) > 0 else 5

        ax.set_ylim(0, max_loss)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels(np.around(normLosses, 3), rotation=45, ha='left')
        ax2.set_xlabel('Losses (mGray)')
        ax.set_ylabel('Losses (mGray)')

        ax.grid(axis='y', alpha=0.3)

        ax.text(7, 1.1 * max_loss / 2, 'Table B', rotation=90, ha='center', va='center', zorder=-1000)
        ax.text(8, 1.1 * max_loss / 2, 'Table C', rotation=90, ha='center', va='center', zorder=-1000)
        ax.text(9, 1.1 * max_loss / 2, 'Beam Dump Front', rotation=90, ha='center', va='center', zorder=-1000)
        ax.text(10.5, 1.1 * max_loss / 2, 'Beam Dump', rotation=90, ha='center', va='center', zorder=-1000)

        [ax.axvline(x - 0.4, color='k', ls='dashed', lw=0.5, alpha=0.5, zorder=-1000) for x in [7, 8, 9, 10]]
        [ax.axvline(x + 0.4, color='k', ls='dashed', lw=0.5, alpha=0.5, zorder=-1000) for x in [7, 8, 9]]

        fig.suptitle(f'TT66 BLMs ({blm_ts}), SC: {cycle_ts}')
        fig.tight_layout()

        # plt.show()
        return fig

    def plot_BLMs_save(self, fname):
        fig = self.plot_BLMs()
        fig.savefig(fname, dpi=200)
        plt.close(fig)

        return fname


class SpsQualityControl(PpmDevice):
    optics_name = 'tt66_2023_FP2_0p25mm_2p2mm'

    def __init__(self, name: str, device_name: str, selector: str = 'SPS.USER.HIRADMT*', japc_field: str = 'INTENSITY.PERFORMANCE'):
        super().__init__(name, device_name, japc_field, selector)

    @primary_callback_wrap
    def callback(self, updater: pyqtSignal, param: str, value: dict, header: dict):
        if 'SPS.USER.HIRADMT' in header['selector']:
            self.optics_name = value['transferLinesOpticsNames']

        #updater.emit()


class VistarCanvas(FigureCanvas):
    FirstUpdate = True
    LegendSettings = {
        'facecolor': 'white',
        'framealpha': 0.5,
        'fontsize': 8,
        'loc': 'upper right',
        'fancybox': False,
        'edgecolor': 'white'
    }

    def __init__(self, img_filter_set=0, width=5, height=4, dpi=100, projection: bool = False, parent=None):
        # WITH GRIDSPEC
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        super(VistarCanvas, self).__init__(fig)

        gs = gridspec.GridSpec(3, 2)

        ax0 = fig.add_subplot(gs[0:2, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])


        self.image = ax0.imshow(np.random.rand(20, 20))
        self.lines = [ax.plot([], [], color='navy', label='Data')[0] for ax in [ax1, ax2]]
        self.fit_lines = [ax.plot([], [], color='red', ls='dashed')[0] for ax in [ax1, ax2]]

        self.axes = [ax0, ax1, ax2, ax3, ax4]
        self.btv_title = self.axes[0].set_title('BTV.XXXX', fontweight='bold')
        self.fig = fig

        title = '\n\n\n\n'
        self.title = self.fig.suptitle(title, fontweight='bold')

        self.img_filter_set = img_filter_set
        self.projection = projection

    def FirstUpdateMod(self):
        # self.cbar_ax = self.fig.add_axes([0.05, 0, 0.05, 1])
        divider = make_axes_locatable(self.axes[0])
        self.cbar_ax = divider.append_axes("left", size="4%", pad=0.7)
        self.cbar = self.fig.colorbar(self.image, cax=self.cbar_ax, aspect=50, label='Fill level (%)')
        self.cbar.ax.yaxis.set_ticks_position('left')
        self.cbar.ax.yaxis.set_label_position('left')
        ticklabels = self.cbar.get_ticks()
        ticklabels = ticklabels/4095 * 100
        ticklabels = [f'{x:.1f}' for x in ticklabels]
        self.cbar.set_ticklabels(ticklabels)

    def format_plot(self):
        self.axes[1].set_title('Horizontal Projection', fontweight='bold')
        self.axes[2].set_title('Vertical Projection', fontweight='bold')
        self.axes[3].set_title('Beam Intensity', fontweight='bold')
        self.axes[4].set_title('Beam Position', fontweight='bold')
        self.axes[0].set_ylabel('Vertical (mm)')
        self.axes[0].set_xlabel('Horizontal(mm)')
        self.axes[1].set_xlabel('Horizontal (mm)')
        self.axes[1].set_ylabel('Amplitude (a.u.)')
        self.axes[2].set_xlabel('Vertical (mm)')
        self.axes[2].set_ylabel('Amplitude (a.u.)')
        self.axes[3].set_xlabel('RF bucket#')
        self.axes[3].set_ylabel('Bunch intensity')
        self.axes[4].set_xlabel('RF bucket#')
        self.axes[4].set_ylabel('Bunch position (mm)')

        for ax in [self.axes[1], self.axes[2], self.axes[4]]:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

        for ax in self.axes[1:3]:
            ax.ticklabel_format(axis='y', scilimits=(-5, 100))
            ax.tick_params(axis='y', labelright=False)

        self.fig.align_ylabels(axs=self.axes[1:])
        self.fig.tight_layout(h_pad=2, rect=[0.01, 0, 0.99, 1])
        # self.fig.subplots_adjust(left=0.1, right=0.9, hspace=0.3)

    def update_title(self, title: str):
        self.title.set_text(title)

    def update_plot(self,
                    h, v, img,
                    # BtvDevice: BtvDigitalCamera,
                    bcts: List[BeamCurrentTransformer],
                    bpms: List[BeamPositionMonitor]):

        ticks = np.linspace(img.min(), img.max(), 5, endpoint=True)

        self.update_imshow(h, v, img)
        self.update_projections(h, v, img)

        self.update_bcts(bcts)
        self.update_bpms(bpms)

        cbar_settings = {
            'ticks': ticks,
            'label': 'Fill level (%)',
            'aspect': 50,
        }

        if self.FirstUpdate:
            self.FirstUpdateMod()
            self.FirstUpdate = False
            return

        self.update_cbar(cbar_settings)

    def update_cbar(self, cbar_settings):
        self.cbar_ax.cla()
        self.cbar = self.fig.colorbar(self.image, cax=self.cbar_ax, **cbar_settings)
        self.cbar.ax.yaxis.set_ticks_position('left')
        self.cbar.ax.yaxis.set_label_position('left')
        ticklabels = self.cbar.get_ticks()
        ticklabels = ticklabels / 4095 * 100
        ticklabels = [f'{x:.1f}' for x in ticklabels]
        self.cbar.set_ticklabels(ticklabels)

    def update_imshow(self, h, v, img):
        extent = (h[0], h[-1], v[-1], v[0])

        self.image.set_data(img)
        self.image.set_extent(extent)
        self.image.autoscale()

    def update_projections(self, h, v, img):
        
        
        if self.projection:
            proj = [np.sum(img, axis=0), np.sum(img, axis=1)]
        else:
            y, x = np.unravel_index(img.argmax(), img.shape)
            proj = [img[y, :], img[:, x]]

        for i, (ax, coordinate, projection) in enumerate(zip(axes, [h, v], proj)):
            lines = ax.lines

            # This line automatically removes the fit line
            if len(lines) > 1:
                lines[-1].remove()

            lines[0].set_data(coordinate, projection)
            ax.relim()
            ax.autoscale_view(True, True, True)
            ax.legend(**self.LegendSettings)

    def add_fit(self, h, v, img) -> dict:
        axes = self.axes[1:3]

        plane = ['Horizontal', 'Vertical']

        fitParams = {}

        if self.projection:
            proj = [np.sum(img, axis=0), np.sum(img, axis=1)]
        else:
            y, x = np.unravel_index(img.argmax(), img.shape)
            proj = [img[y, :], img[:, x]]

        for i, (ax, coordinate, projection) in enumerate(zip(axes, [h, v], proj)):
            ax = axes[i]

            lines = ax.lines
            if len(lines) == 1:
                line = ax.plot([], [], color='red', ls='dashed')[0]
            else:
                line = lines[1]

            try:
                x_fit, y_fit, popt, pcov = get_fit(coordinate, projection, maxfev=100)
                popt_std = np.sqrt(np.diag(pcov))
                fitParams[plane[i]] = (popt, popt_std)

                s = f'{popt[2]:.2f}'
                s_std = f'{popt_std[2]:.2f}'
                c = f'{popt[1]:.2f}'
                c_std = f'{popt_std[1]:.2f}'

                # ToDo: R-squared is not reliably computed (sometimes R2>1?!) so it's not printed anywhere.
                # try:
                #     residuals = y - y_fit
                #     ss_res = np.sum(residuals ** 2)
                #     ss_tot = np.sum((proj - np.mean(proj)) ** 2)
                #     r_squared = 1 - (ss_res / ss_tot)
                # except RuntimeWarning:
                #     r_squared = 0

                label = f'Fit:\nσ={s}$\pm${s_std} mm\n'\
                        f'µ={c}$\pm${c_std} mm\n'
                        # f'R$^2$={r_squared:.3f}'

                line.set_data(x_fit, y_fit)
                line.set_label(label)

            except (ValueError, RuntimeError, OptimizeWarning, np.linalg.LinAlgError) as e:
                fitParams[plane[i]] = None
                line.set_data([], [])
                label = 'Fit Failed!'
                line.set_label(label)

            ax.legend(**self.LegendSettings)
        return fitParams

    def update_bpms(self, bpms: List[BeamPositionMonitor]):
        ax = self.axes[-1]

        all_bpm_sets = [bpm_device.get_bpms() for bpm_device in bpms]

        nlines = 0
        for bpm_sets in all_bpm_sets:
            nlines = nlines + len(bpm_sets)
        nlines = int(nlines*2)

        if len(ax.lines) != nlines:
            [l.remove() for l in ax.get_lines()]
            for bpm_device in bpms:
                for bpm in bpm_device.get_bpms():
                    h, v = bpm_device.get_positions(bpm)
                    lbl_h = f'{bpm[5:]}''.H: µ$_H$='f'{np.mean(h):.1f}$\pm${np.std(h):.1f} mm'
                    lbl_v = f'{bpm[5:]}''.V: µ$_V$='f'{np.mean(v):.1f}$\pm${np.std(v):.1f} mm'
                    ax.plot(h, label=lbl_h)
                    ax.plot(v, label=lbl_v)
        else:
            lines = ax.get_lines()
            i = 0
            for bpm_device in bpms:
                for bpm in bpm_device.get_bpms():
                    h, v = bpm_device.get_positions(bpm)
                    lbl_h = f'{bpm[5:]}''.H: µ$_H$='f'{np.mean(h):.1f}$\pm${np.std(h):.1f} mm'
                    lbl_v = f'{bpm[5:]}''.V: µ$_V$='f'{np.mean(v):.1f}$\pm${np.std(v):.1f} mm'

                    lines[2*i].set_xdata(np.arange(len(h)))
                    lines[2*i].set_ydata(h)

                    lines[2*i+1].set_xdata(np.arange(len(v)))
                    lines[2*i+1].set_ydata(v)
                    lines[2*i].set_label(lbl_h)
                    lines[2 * i + 1].set_label(lbl_v)
                    i = i + 1

        ax.legend(handlelength=0.75, handletextpad=0.5, ncol=2, columnspacing=1.5, **self.LegendSettings)
        ax.relim()
        ax.autoscale()

    def update_bcts(self, bcts: List[BeamCurrentTransformer]):
        ax = self.axes[-2]

        if len(ax.get_lines()) != len(bcts):
            if len(ax.lines) > 0:
                [l.remove() for l in ax.get_lines()]

            for bct in bcts:
                ax.step(np.arange(len(bct.bunchIntensity)), bct.bunchIntensity,
                        label=f'{bct}: {bct.totalIntensity:.2e} ppp', where='mid')
        else:
            lines = ax.get_lines()
            for i, bct in enumerate(bcts):
                lines[i].set_xdata(np.arange(len(bct.bunchIntensity)))
                lines[i].set_ydata(bct.bunchIntensity)
                lines[i].set_label(f'{bct}: {bct.totalIntensity:.2e} ppp')

        ax.legend(**self.LegendSettings)
        ax.relim()
        ax.autoscale()


def print_triggers(ppm_devices: List[Union[BeamCurrentTransformer, BeamPositionMonitor, BeamQualityMonitor]],
                   non_ppm_devices: List[Union[BtvDigitalCamera]],
                   bct_devices: List[BeamCurrentTransformer]):
    if len(ppm_devices) == 0:
        msg = 'No ppm devices!'
        warnings.warn(msg, UserWarning)
        return

    ppm_devices = [device for device in ppm_devices if not type(device) == SpsQualityControl]
    cstamps = [device.cycle_ts for device in ppm_devices]

    # We are dynamically checking the device selectors since we are quite frequently switching users in the SPS
    selectors = [device.selector for device in ppm_devices]
    user = list(set(selectors))[0]
    time_delta = 8 if user == 'SPS.USER.HIRADMT1' else 22

    print(f'{"Device".center(20, " ")}{"Selector".center(20, " ")}'
          f'{"AcqStamp".center(35, " ")}{"Cyclestamp".center(25, " ")}')
    for device in ppm_devices:
        acq_ts = '--' if device.acq_ts == 0 else print_unix_ts(device.acq_ts)
        cycle_ts = '--' if device.cycle_ts == 0 else print_unix_ts(device.cycle_ts)
        print(f'{device.name:^20s}{device.selector:^20s}{acq_ts:^35s}{cycle_ts:^25s}')
    for device in non_ppm_devices:
        acq_ts = '--' if device.acq_ts == 0 else print_unix_ts(device.acq_ts)
        print(f'{device.name:^20s}{device.selector:^20s}{acq_ts:^35s}{"--":^25s}')

    if all([device.totalIntensity > 5e8 for device in bct_devices]):
        print(f'{f"Extraction! TT60: {bct_devices[1].totalIntensity:.2e} / TT66 {bct_devices[0].totalIntensity:.2e}":_^100}')
    else:
        print(f'{f"No Extraction! TT60: {bct_devices[1].totalIntensity:.2e} / TT66 {bct_devices[0].totalIntensity:.2e}":_^100}')

    if len(set(cstamps)) != 1 or any([cstamp is None for cstamp in cstamps]):
        print(f'{"ONE OF YOUR PPM DEVICES TRIGGERED OUTSIDE OF THE LAST CYCLE":!^100}')
        return 'WARNING'
    if any([(device.acq_ts - list(set(cstamps))[0]) > time_delta*1e6 for device in non_ppm_devices]):
        print(f'{"A BTV DID NOT TRIGGER WITH THE BEAM":!^100}')
        return 'WARNING'
    return None


class ExtractionFaker(QObject):
    FirstUpdate = True
    extraction = pyqtSignal()

    nh = 300
    nv = 700
    h = np.linspace(-5, 5, nh)
    v = np.linspace(-8, 8, nv)
    coords = np.array(np.meshgrid(h, v)).T.reshape(-1, 2)

    def create_btv_dataset(self):
        sx = np.random.randint(1, 20) * np.random.rand()
        sy = np.random.randint(1, 20) * np.random.rand()
        mux = np.random.rand() * 3 * 1 if random.random() < 0.5 else -1
        muy = np.random.rand() * 3 * 1 if random.random() < 0.5 else -1
        mu = np.array([mux, muy])

        corr = np.random.rand()
        cov = np.array([[sx**2, corr*sx*sy], [corr*sx*sy, sy**2]])
        while not np.all(np.linalg.eigvals(cov) > 0):
            corr = np.random.rand()
            cov = np.array([[sx ** 2, corr], [corr, sy ** 2]])

        A = np.random.rand() * 4095
        c = np.random.rand() * 1
        img = A*multivariate_normal.pdf(self.coords, mu, cov)+c
        img = img.reshape(self.nh, self.nv).T
        img = np.around(img, 0)

        return self.h, self.v, img

    def run(self, Application: 'HrmtVistar'):
        while True:
            QTest.qWait(5000)

            BtvDevice = Application.BtvDevices[0]
            BctDevice = Application.BctDevices[0]

            h, v, img = self.create_btv_dataset()
            BtvDevice.img_h = h
            BtvDevice.img_v = v
            BtvDevice.img = img

            BctDevice.ExtractionFlag = bool(random.getrandbits(1))
            if BctDevice.ExtractionFlag:
                for device in Application.BctDevices:
                    device.totalIntensity = 1e12
            else:
                for device in Application.BctDevices:
                    device.totalIntensity = 0
            Application.cycle_end()


def get_extraction_event(BctDevices: List[BeamCurrentTransformer],
                         BTV: BtvDigitalCamera,
                         BQM: BeamQualityMonitor,
                         SPSQC: SpsQualityControl,
                         fitParams: dict) -> dict:
    Event = {}

    Event['BCT.cyclestamp'] = BctDevices[0].cycle_ts
    Event['BTV.acqStamp'] = BTV.acq_ts

    Event['User'] = BctDevices[0].selector

    for bct in BctDevices:
        Event[f'{bct.name}.totalIntensity'] = bct.totalIntensity

    Event['SPSQC.optics_name'] = SPSQC.optics_name

    Event['BQM.nbunches'] = BQM.nbunches
    Event['BQM.bunchLengthMean'] = BQM.bunchlength_mean
    Event['BQM.bunchLengthStd'] = BQM.bunchlength_std

    Event['BTV.screen'] = BTV.screen
    Event['BTV.filter'] = BTV.od_filter

    # fitParams
    fitparams_values = get_fitparams(fitParams)
    fitparams_names = ['sx', 'sx_std', 'mux', 'mux_std', 'sy', 'sy_std', 'muy', 'muy_std']
    for name, value in zip(fitparams_names, fitparams_values):
        Event[f'BTV.fit.{name}'] = value

    return Event


def get_optics_name(optics_name: str) -> str:
    # if len(optics_name) > 0:
    #     tfsline = optics_name.split('_')
    #     if len(tfsline) == 4:
    #         tfs_fp = tfsline[2].upper()
    #         tfs_sigma = tfsline[3].replace('p', '.')
    #     else:
    #         tfs_fp = tfsline[2].upper()
    #         tfs_sigma = f"({tfsline[3].replace('p', '.')}mm, {tfsline[4].replace('p', '.')})"
    # else:
    #     tfs_sigma = 'Empty'
    #     tfs_fp = 'Empty'
    #
    # return f'{tfs_fp}, {tfs_sigma}'
    return optics_name


class Logger:
    fpath = '/eos/project-h/hiradmat/HRMT_Data/VISTAR_Log/'
    FirstWrite = True

    def __init__(self):
        fname = f'{datetime.utcnow().isocalendar()[0]}_wk{datetime.utcnow().isocalendar()[1]}.csv'
        self.fpath = self.fpath + fname
        print(self.fpath)
        if os.path.isfile(self.fpath):
            print(f'{self.fpath} exists!')
            self.FirstWrite = False
            return
        print(f'{self.fpath} does not exist. Creating on first extraction!')

    def log(self, Event: dict):
        df = pd.DataFrame.from_dict(Event, orient='index').T

        if not self.FirstWrite:
            print(f'Logging Extraction to {self.fpath}')
            df.to_csv(self.fpath, mode='a', index=False, header=False)
            return

        print(f'Performing First Write to {self.fpath}')
        df.to_csv(self.fpath, mode='a', index=False, header=True)
        self.FirstWrite = False


def get_fitparams(fitParams: dict) -> tuple:
    if fitParams['Horizontal'] is not None:
        popt, popt_std = fitParams['Horizontal']
        sx = popt[2]
        sx_std = popt_std[2]
        mux = popt[1]
        mux_std = popt_std[1]
    else:
        sx = 0
        sx_std = 0
        mux = 0
        mux_std = 0

    if fitParams['Vertical'] is not None:
        popt, popt_std = fitParams['Vertical']
        sy = popt[2]
        sy_std = popt_std[2]
        muy = popt[1]
        muy_std = popt_std[1]
    else:
        sy = 0
        sy_std = 0
        muy = 0
        muy_std = 0

    return sx, sx_std, mux, mux_std, sy, sy_std, muy, muy_std


def get_title(extraction: bool,
              spsqc: SpsQualityControl,
              bqm: BeamQualityMonitor,
              bct: BeamCurrentTransformer,
              acq_ts,
              fit_params=None) -> str:
    if fit_params is not None:
        sx, sx_std, mux, mux_std, sy, sy_std, muy, muy_std = get_fitparams(fit_params)
    tt60_intensity = bct.totalIntensity
    cycle_ts = bct.cycle_ts
    nb = bqm.nbunches
    bl = bqm.bunchlength_mean
    bl_std = bqm.bunchlength_std
    optics_name = get_optics_name(spsqc.optics_name)

    if extraction:
        title = f'Extraction: {tt60_intensity:.2e} ppp, {nb}, ({bl:.2f}±{bl_std:.2f}) ns\n' + \
                f"Supercycle: {print_unix_ts(cycle_ts)}".ljust(50, ' ') + \
                f'{optics_name}'.rjust(50, ' ') + '\n' + \
                f"Acquisition: {print_unix_ts(acq_ts)}".ljust(50, ' ') + \
                f"σ=({sx:.2f}, {sy:.2f}) mm / µ=({mux:.2f}, {muy:.2f}) mm".rjust(50, ' ')
    else:
        title = f'No Extraction :(\n' + \
                f"Supercycle: {print_unix_ts(cycle_ts)}".ljust(50, ' ') + \
                f'{optics_name}'.rjust(50, ' ') + '\n' + \
                f"Acquisition: {print_unix_ts(acq_ts)}".ljust(128, ' ')

    return title


class HrmtVistar(QMainWindow):
    update_signal = pyqtSignal(name='update_signal')
    ExtractionFlag = False

    def extraction_test(self, BtvDevice: BtvDigitalCamera, BctDevice: BeamCurrentTransformer):
        self.thread = QThread()
        self.worker = ExtractionFaker()
        self.worker.moveToThread(self.thread)
        # self.worker.extraction.connect(self.plot_extraction)
        self.thread.started.connect(partial(self.worker.run, self))
        self.thread.start()

    def reset_timer(self):
        self.update_timer.start(500)

    def cycle_end(self):
        cycle_ts = self.BctDevices[0].cycle_ts
        acq_ts = self.BtvDevices[0].acq_ts

        print_triggers(self.cycled_devices, self.non_cycled_devices, self.BctDevices)
        # BTV Business
        BeamTelevision = self.BtvDevices[0]
        h, v, img = BeamTelevision.get_img()
        if self.canvas.img_filter_set > 0:
            img = img_filter(img, self.canvas.img_filter_set)

        self.canvas.btv_title.set_text(
            f'{self.BtvDevices[0].name} ({self.BtvDevices[0].screen}/{self.BtvDevices[0].od_filter})'
        )
        self.canvas.update_plot(h, v, img, self.BctDevices, self.BpmDevices)

        self.ExtractionFlag = self.BctDevices[0].ExtractionFlag

        if not self.ExtractionFlag:
            title = f'No Extraction :(\n' + \
                    f"Supercycle: {print_unix_ts(cycle_ts)}".ljust(50, ' ') + \
                    f'{get_optics_name(self.SPSQC.optics_name)}'.rjust(50, ' ') + '\n' + \
                    f"Acquisition: {print_unix_ts(acq_ts)}".ljust(128, ' ')
            title = get_title(extraction=self.ExtractionFlag,
                              spsqc=self.SPSQC,
                              bqm=self.BQM,
                              bct=self.BctDevices[0],
                              acq_ts=acq_ts
                              )
            self.canvas.update_title(title)
            self.canvas.draw()
            print('\n\n\n')
            return

        fitParams = self.canvas.add_fit(h, v, img)
        title = get_title(extraction=self.ExtractionFlag,
                          spsqc=self.SPSQC,
                          bqm=self.BQM,
                          bct=self.BctDevices[0],
                          acq_ts=acq_ts,
                          fit_params=fitParams
                          )
        self.canvas.update_title(title)
        self.canvas.draw()

        if self.logbook_enabled or self.logging_enabled:
            Extraction = get_extraction_event(
                self.BctDevices,
                self.BtvDevices[0],
                self.BQM,
                self.SPSQC,
                fitParams
            )

        if self.logbook_enabled:
            cycle_ts = Extraction['BCT.cyclestamp']
            vistar_screenshot = f'/eos/project-h/hiradmat/HRMT_Data/VISTAR_Log/VISTAR_Screenshots/HRMT_VISTAR_{cycle_ts}.png'
            self.canvas.fig.savefig(vistar_screenshot, dpi=200)
            self.send_to_logbook(Extraction, fitParams, self.BLM, vistar_screenshot)

        if self.logging_enabled:
            self.logger.log(Extraction)

        print('\n\n\n')

    def init_logbook(self):
        auth_client = pyrbac.AuthenticationClient.create()
        token = auth_client.login_explicit(self.user, self.pwd)

        client = pylogbook.Client(rbac_token=token)
        logbook = client.get_activities()['HiRadMat']
        tags = list(pylogbook.Client(rbac_token=token).get_tags())
        [tag] = [tag for tag in tags if tag.name == 'HiRadMat_extraction']

        logbook = pylogbook.ActivitiesClient(
            activities=logbook,
            client=client,
        )

        return token, logbook, tag

    def send_to_logbook(self, ExtractionEvent: dict, fitParams: dict, BLM: BeamLossMonitor, fname: str):
        btv_ts = ExtractionEvent['BTV.acqStamp']
        cycle_ts = ExtractionEvent['BCT.cyclestamp']
        optics_name = get_optics_name(ExtractionEvent['SPSQC.optics_name'])
        nbunches = ExtractionEvent['BQM.nbunches']
        bl = ExtractionEvent['BQM.bunchLengthMean']
        bl_std = ExtractionEvent['BQM.bunchLengthStd']

        TT60_totalIntensity = ExtractionEvent['TT60.BCT.610225.totalIntensity']
        btv_screen = ExtractionEvent['BTV.screen']
        btv_filter = ExtractionEvent['BTV.filter']
        sx, sx_std, mux, mux_std, sy, sy_std, muy, muy_std = get_fitparams(fitParams)

        line1 = '<table style="width: 100%;"><tbody>' \
                '<tr><td style="text-align: left;">' \
                f'<strong><span style="font-size: 24px">HiRadMat Extraction<br>{optics_name}<br>{print_unix_ts(btv_ts)}</span></strong>' \
                f'</td><td style="text-align: right;"><strong><span style="font-size: 24px">' \
                f'{TT60_totalIntensity:.2e} protons ({nbunches} b)'

        if sx == 0 or sy == 0:
            line2 = '<br><span style="color: rgb(255, 0, 0);">BTV PROFILE FIT ERROR</span>' \
                    f'</span></strong></td></tr></tbody></table>'
            line5 = '<br><strong>BTV Beam sigma and position measurement:</strong>\n' \
                    f'Screen type: {btv_screen}\n' \
                    f'Filter setting: {btv_filter}\n' \
                    f'<span style="color: rgb(255, 0, 0);">BTV PROFILE FIT ERROR</span>'
        else:
            line2 = f'''<br>sigma: x={sx:.2f} mm, y={sy:.2f} mm<br>position: x={mux:.2f} mm, y={muy:.2f} mm</span></strong></td>
        </tr></tbody></table>'''
            line5 = f'''<br>BTV Beam sigma and position measurement: <strong>{btv_screen} / {btv_filter}</strong>
        <table style="width: 100%;"><tbody><tr>
        <td></td>
        <td colspan="2" style="text-align: center;"><strong>Horizontal</strong></td>
        <td colspan="2" style="text-align: center;"><strong>Vertical</strong></td>
        </tr><tr>
        <td style="text-align: right;"><strong>Spot size, sigma (mm)</strong></td>
        <td style="text-align: center;">{sx:.2f}</td>
        <td style="text-align: center;">{sx_std:.2f}</td>
        <td style="text-align: center;">{sy:.2f}</td>
        <td style="text-align: center;">{sy_std:.2f}</td>
        </tr><tr>
        <td style="text-align: right;"><strong>position,  (mm)</strong></td>
        <td style="text-align: center;">{mux:.2f}</td>
        <td style="text-align: center;">{mux_std:.2f}</td>
        <td style="text-align: center;">{muy:.2f}</td>
        <td style="text-align: center;">{muy_std:.2f}</td>
        </tr></tbody></table>'''
        line12 = line1 + line2

        line3 = f'Cycle timestamp: {print_unix_ts(cycle_ts)} ({cycle_ts:.0f})<br />' \
                f'Acq timestamp: {print_unix_ts(btv_ts)} ({btv_ts:.0f})<br />'
        line4 = f'Average bunch length: <strong>{bl * 1e9:.2f}±{bl_std * 1e9:.2f} ns</strong><br />'

        if len(list(BLM.normLosses.keys())) > 0:
            BLM_losses = list(BLM.normLosses.values())
            line6 = f'''<br><strong>Beam Loss Monitors:</strong>
                    <table style="width: 100%;"><tbody><tr>
                    <td>BLMH.660042</td><td>{BLM_losses[0]:.4}</td><td>BLMH.660518</td><td>{BLM_losses[6]:.4}</td><td>TT66 Vacuum Window</td>
                    </tr><tr>
                    <td>BLMH.660122</td><td>{BLM_losses[1]:.4}</td><td>BLMH.660523</td><td>{BLM_losses[7]:.4}</td><td>Table B</td>
                    </tr><tr>
                    <td>BLMH.660225</td><td>{BLM_losses[2]:.4}</td><td>BLMH.660526</td><td>{BLM_losses[8]:.4}</td><td>Table C</td>
                    </tr><tr>
                    <td>BLMH.660308</td><td>{BLM_losses[3]:.4}</td><td>BLMH.660530</td><td>{BLM_losses[9]:.4}</td><td>Beam Dump Front</td>
                    </tr><tr>
                    <td>BLMH.660411</td><td>{BLM_losses[4]:.4}</td><td>BLMH.660531</td><td>{BLM_losses[10]:.4}</td><td rowspan="2">Beam Dump</td>
                    </tr><tr>
                    <td>BLMH.660511</td><td>{BLM_losses[5]:.4}</td><td>BLMH.660535</td><td>{BLM_losses[11]:.4}</td>
                    </tr></tbody></table>'''
        else:
            line6 = ''

        logbook_lines = [line12, line3, line4, line5, line6]

        logbook_text = ''
        for line in logbook_lines:
            logbook_text = logbook_text + line

        try:
            event = self.logbook.add_event(logbook_text, tags=self.tag)
            event.attach_file(fname)
        except pylogbook.exceptions.AuthenticationError:
            self.token, self.logbook, self.tag = self.init_logbook()
            event = self.logbook.add_event(logbook_text, tags=self.tag)
            event.attach_file(fname)

        if len(list(BLM.normLosses.keys())) > 0:
            fname = f'/eos/project-h/hiradmat/HRMT_Data/VISTAR_Log/VISTAR_BLMs/HRMT_BLM_{cycle_ts}.png'
            event.attach_file(BLM.plot_BLMs_save(fname))

    def __init__(self,
                 user: str,
                 pwd: str,
                 img_filter_set: int = 3,
                 logbook_enabled: bool = False,
                 logging_enabled: bool = False,
                 test_vistar: bool = False):
        super(HrmtVistar, self).__init__()

        self.user = user
        self.pwd = pwd

        # Initialize logging
        if logging_enabled:
            self.logger = Logger()
        self.logging_enabled = logging_enabled

        # Initialize logbook
        if logbook_enabled:
            print('Logbook logging activated...')
            self.token, self.logbook, self.tag = self.init_logbook()
            print('...connected to e-Logbook!')
        self.logbook_enabled = logbook_enabled

        # Timer Function
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.cycle_end)

        # If update
        # The update_signal function connects the callbacks (which are executed in a different thread) to the main thread.
        self.update_signal.connect(self.reset_timer)

        title = f'HiRadMat VISTAR - Logging {"ON" if logging_enabled else "OFF"} - Logbook {"ON" if logbook_enabled else "OFF"}'
        self.setWindowTitle(title)

        self.main_frame = QWidget()

        dpi = 100
        self.canvas = VistarCanvas(
            img_filter_set=img_filter_set,
            projection=True,
            width=int(1000/dpi),
            height=int(1280/dpi),
            dpi=dpi)
        self.canvas.setParent(self.main_frame)

        mainlayout = QVBoxLayout()
        mainlayout.addWidget(self.canvas)
        self.main_frame.setLayout(mainlayout)
        self.setCentralWidget(self.main_frame)

        self.canvas.fig.tight_layout()
        self.canvas.draw()
        self.canvas.format_plot()

        self.BtvDevices = [
            BtvDigitalCamera(
                name='TT66.BTV.660524',
                device_name='TT66.BTV.660524.DigiCam',
                filter_channel='TT66.BTV.660524.Filter',
                screen_channel='TT66.BTV.660524.Screen'
            ),
            #BtvICamera(
            #    name='BTV.660518',
            #    device_name='TT66.BTV.660518'
            #)
        ]
        self.BctDevices = [
            BeamCurrentTransformer(
                name='TT66.BCT.660289',
                device_name='TT66.BCTFI.660298'
            ),
            BeamCurrentTransformer(
                name='TT60.BCT.610225',
                device_name='TT60.BCTFI.610225'
            )
        ]
        self.BpmDevices = [
            BeamPositionMonitor(
                name='TT66.BPMs',
                device_name='BPMITT66',
                bpms=['TT66.BPM.660517', 'TT66.BPKG.660529']
                # bpms=['660517', '660524', '660529']
            )
        ]
        self.BQM = BeamQualityMonitor(
            name='SPS.BQM',
                device_name='SPSBQMSPSv1'
        )
        self.SPSQC = SpsQualityControl(
            name='SPSQC',
            device_name='SPSQC'
        )
        self.BLM = BeamLossMonitor(
            name='TT66.BLMs',
            device_name='BLMITI2UP'
        )

        self.canvas.btv_title.set_text(
            f'{self.BtvDevices[0].name} ({self.BtvDevices[0].screen}/{self.BtvDevices[0].od_filter})'
        )

        devices = self.BtvDevices + self.BctDevices + self.BpmDevices + [self.BQM, self.SPSQC, self.BLM]

        # For later we need to get all cycle-bound and non cycle-bound devices separately
        ix = np.array([issubclass(device.__class__, PpmDevice) for device in devices])
        self.cycled_devices = [devices[i] for i in range(len(devices)) if ix[i]]
        self.non_cycled_devices = [devices[i] for i in range(len(devices)) if ~ix[i]]

        if test_vistar:
            self.extraction_test(self.BtvDevices[0], self.BctDevices[0])
            return

        japc = pyjapc.PyJapc(incaAcceleratorName="SPS", selector='SPS.USER.HIRADMT*')

        shs = []
        for device in devices:
            shs = shs + device.create_subscriptions(japc, self.update_signal)

        for sh in shs:
            sh.startMonitoring()

        self.devices = devices
        self.japc = japc
        self.shs = shs

    def closeEvent(self, event):
        """
        This is executed when you close the window your App is running in.
        """
        print('You closed the window...')
        self.stopSubscriptions()
        event.accept()

    def __del__(self):
        """
        This is executed when init fails or the module/script is killed in the terminal.
        """
        print('Init has aborted...')
        self.stopSubscriptions()

    def stopSubscriptions(self):
        if hasattr(self, 'shs'):
            if self.shs:
                [sh.stopMonitoring() for sh in self.shs]


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-user')
    args.add_argument('-pwd')
    args.add_argument('-logbook', default=False, action='store_true')
    args.add_argument('-logging', default=False, action='store_true')
    args.add_argument('-test', default=False, action='store_true')
    args.add_argument('-img_filter', default=3, type=int)
    argv = args.parse_args()
    if argv.logbook:
        if (argv.user is None) | (argv.pwd is None):
            print('When using the logbook option, you need to provide a username (-pwd=USERNAME) and password (-pwd=PASSWORD)')
            return

    app = QApplication([])
    win = HrmtVistar(user=argv.user,
                     pwd=argv.pwd,
                     img_filter_set=argv.img_filter,
                     logbook_enabled=argv.logbook,
                     logging_enabled=argv.logging,
                     test_vistar=argv.test)
    win.show()
    sys(app.exec_())


if __name__ == '__main__':
    main()
