#!/usr/bin/env python3
#
#  Since: June, 2021
#  Author: gvenzl
#  Name: ImageSketcher.py
#  Description: The image sketcher program
#
#  Copyright (c) 2021 Gerald Venzl
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#

from flask import Flask, send_file
from flask_restful import Resource, Api, reqparse, abort
from enum import Enum
import werkzeug.datastructures
import numpy
import cv2
import io

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
# Add type argument to URL (https://..../?type=...)
parser.add_argument("type", location="args",
                    default="SKETCHED")
# Accept image as binary file memory based storage
parser.add_argument("image",
                    type=werkzeug.datastructures.FileStorage,
                    location="files",
                    required=True,
                    help="Please provide a file.")


# Little enum to help with types
class Mode(Enum):
    GRAYSCALE = "grayscale"
    GRAYSCALE_INVERTED = "grayscale_inverted"
    SKETCHED = "sketched"


class Sketch(Resource):
    # Define POST request
    @staticmethod
    def post():
        # Parse arguments
        args = parser.parse_args()
        # Get conversion mode (defaults to sketched as per parser.add_argument(default=))
        # Catch conversion error for when someone passes in an invalid type
        try:
            mode = Mode[args["type"].upper()]
        except KeyError as e:
            # Return HTTP 400 with message
            abort(400, message="Choose 'type' from 'grayscale', 'grayscale_inverted', 'sketched'.")
            return None
        # Read file as a stream
        file_stream = args["image"].read()
        # Sketch the image
        image = sketch_image(file_stream, mode)
        # Return binary response with mimetype 'image/jpeg'
        return send_file(io.BytesIO(image), mimetype='image/jpeg')

    @staticmethod
    def get():
        # Return 405 message to reject GET call
        abort(405, message="Please send image as POST request.")
        return


# Accept data on / and /sketch
api.add_resource(Sketch, '/', "/sketch")


def sketch_image(image, mode):

    # Create new image version image
    image = create_image(image, mode)

    # Encode image for transport
    image_encode = cv2.imencode(".jpg", image)[1]

    # Return image bytes
    return image_encode.tobytes()


def create_image(file_stream, mode):

    # Convert file stream to Numpy array
    np_img = numpy.frombuffer(file_stream, numpy.uint8)
    # Decode image from numpy array
    image = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode is Mode.GRAYSCALE:
        return image_gray

    # Invert grayscale image
    image_invert = cv2.bitwise_not(image_gray)
    if mode is Mode.GRAYSCALE_INVERTED:
        return image_invert

    # Blur grayscale inverted image
    # Param 1: image
    # Param 2: Gaussian kernel size
    # Param 3: Gaussian kernel standard deviation in X direction
    # Param 4: Gaussian kernel standard deviation in Y direction
    image_blurred = cv2.GaussianBlur(image_invert, (21, 21), sigmaX=0, sigmaY=0)

    # Subtract grayscale from blurred image
    sketched_image = dodge(image_gray, image_blurred)
    if mode is Mode.SKETCHED:
        return sketched_image


def dodge(x, y):
    # Divide one array by another
    return cv2.divide(x, 255 - y, scale=256)


# Run app
if __name__ == '__main__':
    # For dev environment
    app.run(None, 8080)
