#!/usr/bin/env python

"""
Helper functions used by the rgbdriverkit
"""

__author__ = 'RGB Photonics GmbH'

def enum(**enums):
    return type('Enum', (), enums)
