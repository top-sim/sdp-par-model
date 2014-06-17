#!/usr/bin/env bash
awk -f dump_maxima_code.awk MajCycleModel.tex    > MajCycleModel.wxm
awk -f dump_maxima_code.awk DataFlow.tex         > DataFlow.wxm
awk -f dump_maxima_code.awk CompHWPowerModel.tex > CompHWPowerModel.wxm
