#
# Since: June, 2021
# Author: gvenzl
# Name: Dockerfile
# Description: Dockerfile to build a little Image Sketcher container image
#
# Copyright 2021 Gerald Venzl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM oraclelinux:8-slim

RUN microdnf install python3 libglvnd-glx

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY ImageSketcher.py .

CMD [ "python3", "./ImageSketcher.py" ]