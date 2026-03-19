#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global setting to enable/disable pytorch compiler
"""
import numpy as np 
disable_compile=False

def change_compile_setting(disable):
    global disable_compile
    disable_compile = disable