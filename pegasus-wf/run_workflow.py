#!/usr/bin/env python3
# coding: utf-8

import glob 
import os
import numpy as np
from Pegasus.api import *
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
props = Properties()
props["pegasus.mode"] = "development"
props.write()

### ADD INPUT FILES TO REPILCA CATALOG
rc = ReplicaCatalog()
