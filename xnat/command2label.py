#!/usr/bin/env python
'''
Command to Label

Usage: command2label <command>...
'''


import re
import sys
import json

argsList = sys.argv[1:]

commandStrList = []
for commandFile in argsList:
    with open(commandFile) as f:
        commandObj = json.load(f)
    commandStr = json.dumps(commandObj) \
                        .replace('"', r'\"') \
                        .replace('$', r'\$')
    commandStrList.append(commandStr)

print('LABEL org.nrg.commands="[{}]"'.format(', \\\n\t'.join(commandStrList)))
