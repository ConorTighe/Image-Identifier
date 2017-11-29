from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import json
from flask import Flask, render_template, request, jsonify
from PIL import Image
import requests
import tensorflow as tf
import numpy as np
from six.moves import urllib

app = Flask(__name__)

FLAGS = None

# Get the current directory
cwd = os.getcwd();

# This handles the graph for our digits
def create_graph():
  # Creates graph from saved output.pb for our digits
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'output.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    
# This handles the graph for the Inception-v3 1000 class dataset
def create_graph_Imgs():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    
# This is the NodeLookup class based on the Inception-v3 example on Github from tensorflow but ive customized it to work with a flask app, NodeLook up is to make naviaging the large dataset managable
class NodeLookup(object):
  # Set up our flags
  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  # Load up our tensors and convert them to readable english
  def load(self, label_lookup_path, uid_lookup_path):
    # Check for our files
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
    # Return node
    return node_id_to_name
  # Convert int id to string
  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

# This is our regular run interface for checking digits
def run_inference_on_image(image):
  # Check if files exist
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Loads label file, strips off carriage return
  label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("./labels.txt")]
    
  # Creates graph from saved GraphDef.
  create_graph()

  # Start tensor sessionn
  with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # Store predictions of image data generated with softmax tensor
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    # output nodes to cmd 
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
        
    # return nodes to post def
    return [(label_lines[node_id], float(predictions[0][node_id])) for node_id in top_k]

# This runs the image interface of normal images
def run_inference_on_image_full(image):
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph_Imgs()

  # Get the labels
  label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("./imagenet_synset_to_human_label_map.txt")]

  # Start tensor session
  with tf.Session() as sess:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    
    # Remove single-dimensional entries from predictions
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    # Prepare predictions for returning
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
    # Return prediction
    return [(label_lines[node_id], float(predictions[node_id])) for node_id in top_k]

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Handle digit posts
@app.route('/uploader', methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        Result = image_handler(f.filename)
        return render_template('results.html', response = Result)
    else:
        return """<html><body>
        Something went horribly wrong
        </body></html>"""

# Handle image posts
@app.route('/imgUploader', methods = ['POST'])
def upload_file_full():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        Result = image_handler_ImageNet(f.filename)
        return render_template('results.html', response = Result)
    else:
        return """<html><body>
        Something went horribly wrong
        </body></html>"""

# Apply tensorflow DNN to digit provided
def image_handler(fname):
    image = (FLAGS.image_file if FLAGS.image_file else
      os.path.join(FLAGS.model_dir, fname))
    print("Image: ", image)
    predictions = run_inference_on_image(image)
    print("pridictions" , predictions)
    return predictions

# Apply tensorflow DNN to image provided
def image_handler_ImageNet(fname):
    image = (FLAGS.image_file if FLAGS.image_file else
      os.path.join(FLAGS.model_dir, fname))
    print("Image: ", image)
    predictions = run_inference_on_image_full(image)
    print("pridictions" , predictions)
    return predictions
    
def main(_):
    app.run()
    
# Sets up the flags for mdoel directory, image file and the amount of predictions returned.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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