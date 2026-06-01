"""Raw TFRecord read throughput for a list of shard files (no parse, no GPU)."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
files = sys.argv[1:]
ds = tf.data.TFRecordDataset(files, num_parallel_reads=8)
t0 = time.time(); nbytes = 0
for rec in ds:
    nbytes += len(rec.numpy())
dt = time.time() - t0
print(f"{len(files)} shards: {nbytes/1e9:.2f} GB in {dt:.1f}s = {nbytes/1e6/dt:.0f} MB/s")
