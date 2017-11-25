from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
from flask import Flask, render_template, request, jsonify
from PIL import Image
import requests
import tensorflow as tf
import numpy as np
from six.moves import urllib

app = Flask(__name__)

FLAGS = None

cwd = os.getcwd();

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'output.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Loads label file, strips off carriage return
  label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("./labels.txt")]
    
  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    
    return [(label_lines[node_id], float(predictions[0][node_id])) for node_id in top_k]

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploader', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        jsonResult = image_handler(f.filename)
        return jsonResult
    else:
        return """<html><body>
        Something went horribly wrong
        </body></html>"""

def image_handler(fname):
    image = (FLAGS.image_file if FLAGS.image_file else
      os.path.join(FLAGS.model_dir, fname))
    print("Image: ", image)
    predictions = run_inference_on_image(image)
    print("pridictions" , predictions)
    return jsonify(predictions=predictions)

def main(_):
    app.run()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
      '--model_dir',
      type=str,
      default=cwd,
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
      '--image_file',
      type=str,
      default='',
      help='Absolute path to image file.'
    )
    parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=5,
      help='Display this many predictions.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)