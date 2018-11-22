# About the dataset in this folder
'face2glyphs.py' produced the files found in ./bez and ./img\*,
where the number after image indicates the size of the image in pixels

the files in bez folder are pickled lists of [4xMx2] numpy arrays, each of which encodes
the bezier curves of a glyph of the font (which appears in the filename). The glpyhs 
encoded are 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', in that order. So the defined glyph can be implicitly determined by its position in the list.

the files in the img\* folder are pickled lists of [\*,\*] numpy arrays which represent
rasterized images of the font's glyphs, in the same order as described for the bez files above.
