# Castor and Sean

Castor and Sean is an experiment in OpenCV-based face matching.

Face matching is a relaxed form of face recognition: it will always
recognize a face as someone from the database.

## castor

castor is a command line tool for face database preprocesing.

castor uses OpenCV (and boost and gflags)

Usage:

    castor DATABASE [--mode=MODE] [MODE SPECIFIC SWITCHES]

the `DATABASE` is just a directory tree. Inside the top directory
different operation modes create or expect to find different
subdirectories.

`MODE` is an operation that castor will perform. Default mode is import.

### --mode=import

This mode expects `DATABASE/new` to contain incoming images. Castor
will try to find faces in the images and cut them out, convert to
greyscale, stretch contrast and store the cutout face in
`DATABASE/seed`.

### --mode=pca

This mode performs Principal Component Analysis on faces from
`DATABASE/seed`. The result is a set of eigenfaces stored in
`DATABASE/eigen`. The image files in eigen are for demonstration only,
they are postprocessed and lost some precision. The actual result of PCA
is in the file `DATABASE/eigen/pca.yml`.

### --mode=pca

This mode projects faces found in `DATABASE/seed` into eigenspace and
stores the results in `DATABASE/projection/projection.yml`. This file
contains a list of filenames and a matrix with the results of projection
(one projected vector per row).

## sean

Sean is a visualisation of the faces projection in 3-dimensional space
of the first three eigenvectors.

Sean uses cinder (and hence has to be 32-bit) and the 32-bit versions of
opencv and boost that come with cinder. At the moment sean is mac only
(windows port would require slight adjustment of the resources handling
and the build system).
