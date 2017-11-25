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
    
def create_graph_Imgs():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    
class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, cwd)
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, cwd)
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


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

def run_inference_on_image_full(image):
    tf.Session()
    print("Tensorflow session ready")
    node_lookup = NodeLookup()
    print("Node lookup loaded")
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # sort the predictions
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

    # map to the friendly names and return the tuples
    return [(node_lookup.id_to_string(node_id), float(predictions[node_id])) for node_id in top_k]

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
    
@app.route('/imgUploader', methods = ['POST'])
def upload_file_full():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        jsonResult = image_handler_ImageNet(f.filename)
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

def image_handler_ImageNet(fname):
    image = (FLAGS.image_file if FLAGS.image_file else
      os.path.join(FLAGS.model_dir, fname))
    print("Image: ", image)
    predictions = run_inference_on_image_full(image)
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