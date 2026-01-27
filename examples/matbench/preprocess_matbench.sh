#!/bin/bash
#matbench_mp_is_metal classification task that also has structures
#for task in matbench_dielectric matbench_jdft2d matbench_log_gvrh matbench_log_kvrh matbench_mp_e_form matbench_mp_gap matbench_perovskites matbench_phonons; do
#for task in matbench_dielectric; do
#for task in matbench_log_gvrh; do
#for task in matbench_log_kvrh; do
#for task in matbench_mp_e_form; do
#for task in matbench_mp_gap; do
#for task in matbench_perovskites; do
#for task in matbench_phonons; do
for task in matbench_jdft2d; do
  echo "Processing $task"
  for fold in {0..4}; do
    echo "Processing fold $fold"
    python examples/matbench/matbench_preonly.py --task_name $task --fold $fold --modelname ${task}_${fold}
  done
done
