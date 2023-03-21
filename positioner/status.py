#!/usr/bin/env python
# coding: utf-8

import ctypes

uint32_t = ctypes.c_uint32

__all__ = ['BootloaderStatus', 'PositionerStatus']


class BootloaderFlagBits(ctypes.LittleEndianStructure):
    """bootloader status register flags"""
    _fields_ = [
                ('init', uint32_t, 1),                  # 0x00000001
                ('timeout', uint32_t, 1),               # 0x00000002
                ('reserve2', uint32_t, 1),              # 0x00000004
                ('reserve3', uint32_t, 1),              # 0x00000008
                ('reserve4', uint32_t, 1),              # 0x00000010
                ('reserve5', uint32_t, 1),              # 0x00000020
                ('reserve6', uint32_t, 1),              # 0x00000040
                ('reserve7', uint32_t, 1),              # 0x00000080
                ('config_changed', uint32_t, 1),        # 0x00000100
                ('bsettings_changed', uint32_t, 1),     # 0x00000200
                ('reserve10', uint32_t, 1),             # 0x00000400
                ('reserve11', uint32_t, 1),             # 0x00000800
                ('reserve12', uint32_t, 1),             # 0x00001000
                ('reserve13', uint32_t, 1),             # 0x00002000
                ('reserve14', uint32_t, 1),             # 0x00004000
                ('reserve15', uint32_t, 1),             # 0x00008000
                ('receiving_firmware', uint32_t, 1),    # 0x00010000
                ('reserve17', uint32_t, 1),             # 0x00020000
                ('reserve18', uint32_t, 1),             # 0x00040000
                ('reserve19', uint32_t, 1),             # 0x00080000
                ('reserve20', uint32_t, 1),             # 0x00100000
                ('reserve21', uint32_t, 1),             # 0x00200000
                ('reserve22', uint32_t, 1),             # 0x00400000
                ('reserve23', uint32_t, 1),             # 0x00800000
                ('firmware_received', uint32_t, 1),     # 0x01000000
                ('firmware_ok', uint32_t, 1),           # 0x02000000
                ('firmware_bad', uint32_t, 1),          # 0x04000000
                ('reserve27', uint32_t, 1),             # 0x08000000
                ('reserve28', uint32_t, 1),             # 0x10000000
                ('reserve29', uint32_t, 1),             # 0x20000000
                ('reserve30', uint32_t, 1),             # 0x40000000
                ('reserve31', uint32_t, 1)]             # 0x80000000


class BootloaderStatus(ctypes.Union):
    """"bootloader status"""
    _anonymous_ = ('bit',)
    _fields_ = [
                ('bit', BootloaderFlagBits),
                ('asInt', uint32_t)]


class PositionerFlagBits(ctypes.LittleEndianStructure):
    """positioner status register flags"""
    _fields_ = [
        ('init', uint32_t, 1),
        ('config_changed', uint32_t, 1),
        ('bsettings_changed', uint32_t, 1),
        ('reserve3', uint32_t, 1),
        ('reserve4', uint32_t, 1),
        ('reserve5', uint32_t, 1),
        ('reserve6', uint32_t, 1),
        ('reserve7', uint32_t, 1),
        ('receiving_trajectory', uint32_t, 1),
        ('trajectory_alpha_received', uint32_t, 1),
        ('trajectory_beta_received', uint32_t, 1),
        ('reserve11', uint32_t, 1),
        ('reserve12', uint32_t, 1),
        ('reserve13', uint32_t, 1),
        ('reserve14', uint32_t, 1),
        ('reserve15', uint32_t, 1),
        ('position_error_overvalue', uint32_t, 1),
        ('reserve17', uint32_t, 1),
        ('reverse_mode', uint32_t, 1),
        ('go_to_datum_alpha', uint32_t, 1),
        ('go_to_datum_beta', uint32_t, 1),
        ('datum_initialization', uint32_t, 1),
        ('datum_alpha_initialized', uint32_t, 1),
        ('datum_beta_initialized', uint32_t, 1),
        ('displacement_completed', uint32_t, 1),
        ('alpha_displacement_completed', uint32_t, 1),
        ('beta_displacement_completed', uint32_t, 1),
        ('alpha_collision', uint32_t, 1),
        ('beta_collision', uint32_t, 1),
        ('datum_initialized', uint32_t, 1),
        ('estimated_position', uint32_t, 1),
        ('position_restored', uint32_t, 1)]


class PositionerStatus(ctypes.Union):
    """positioner status register"""
    _anonymous_ = ('bit',)
    _fields_ = [
        ('bit', PositionerFlagBits),
        ('asInt', uint32_t)]

