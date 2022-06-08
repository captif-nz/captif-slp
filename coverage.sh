#!/bin/bash

pytest -Werror --cov-report xml:./cov.xml --cov captif_slp -v -m "not slow" tests
rm .coverage*
