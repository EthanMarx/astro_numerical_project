import matplotlib.pyplot as plt
import readsnapGadget2 as snap
import numpy as np

snapnum = 95
filename = f"data/snapshot_{str(snapnum).zfill(3)}"

head = snap.snapshot_header(filename)
redshift = head.redshift

pos = snap.read_block(filename, "POS ")
mass = snap.read_block(filename, "MASS")




