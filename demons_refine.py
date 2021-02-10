#!/usr/bin/env python


from __future__ import print_function

import SimpleITK as sitk
import sys
import os
import pandas as pd


import numpy as np

def command_iteration(filter):
    print("{0:3} = {1:10.5f}".format(filter.GetElapsedIterations(),
                                     filter.GetMetric()))
    
'''
def demons_registration(fixed, moving, ini_def):
  registration_method = sitk.ImageRegistrationMethod()
  transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
  transform_to_displacement_field_filter.SetReferenceImage(fixed)
  initial_transform = sitk.DisplacementFieldTransform(ini_def)


'''
'''if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "<fixedImageFilter> <movingImageFile>",
          "[initialTransformFile] <outputTransformFile>")
    sys.exit(1)'''


def refine_def(fixed, moving, ini_def, num):

    transformDomainMeshSize = [8] * moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,
                                      transformDomainMeshSize)

    print("Initial Parameters:")
    print(tx.GetParameters())

    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(250)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(2)
    #demons.SetMovingInitialTransform(ini_def)
    #demons.SetInitialTransform(ini_def, True)
    ini_def = sitk.DisplacementFieldTransform(ini_def)

    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))
    toDisplacementFilter = sitk.TransformToDisplacementFieldFilter()
    toDisplacementFilter.SetReferenceImage(fixed)
    displacementField = toDisplacementFilter.Execute(ini_def)


    displacementField = demons.Execute(fixed, moving, displacementField)

    print("-------")
    print("Number Of Iterations: {0}".format(demons.GetElapsedIterations()))
    print(" RMS: {0}".format(demons.GetRMSChange()))

    outTx = sitk.DisplacementFieldTransform(displacementField)

    return outTx

