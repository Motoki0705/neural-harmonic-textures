#!/usr/bin/env bash
set -euo pipefail

input_video="${1:?usage: build_sfm_benchmarks.sh INPUT_VIDEO OUTPUT_DIR}"
output_dir="${2:?usage: build_sfm_benchmarks.sh INPUT_VIDEO OUTPUT_DIR}"
mkdir -p "${output_dir}"

ffmpeg -y -hide_banner -loglevel warning -ss 120 -t 60 -i "${input_video}" \
  -vf "tmix=frames=5:weights='1 1 1 1 1',eq=brightness=-0.12:contrast=1.15" \
  -an -c:v libx264 -preset medium -crf 18 "${output_dir}/tennis-court-blur-lowlight.mp4"

ffmpeg -y -hide_banner -loglevel warning -ss 240 -t 60 -i "${input_video}" \
  -vf "zoompan=z='1+0.18*sin(on/240)*sin(on/240)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=60000/1001" \
  -an -c:v libx264 -preset medium -crf 18 "${output_dir}/tennis-court-zoom.mp4"

ffmpeg -y -hide_banner -loglevel warning -ss 360 -t 60 -i "${input_video}" \
  -an -c:v libx264 -preset medium -crf 18 \
  "${output_dir}/tennis-court-dynamic-control.mp4"

ffmpeg -y -hide_banner -loglevel warning -ss 360 -t 60 -i "${input_video}" \
  -vf "drawbox=x='mod(t*220\,w+300)-150':y='h*0.30':w=110:h=360:color=black@0.90:t=fill,drawbox=x='w-mod(t*150\,w+260)':y='h*0.45':w=80:h=280:color=white@0.85:t=fill" \
  -an -c:v libx264 -preset medium -crf 18 "${output_dir}/tennis-court-dynamic-occlusion.mp4"
