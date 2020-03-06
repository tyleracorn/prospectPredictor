.. _predictor:
.. public

Predictor 
#########

the predictor class(s) (only 1 currently) is a class for predicting a prospectivity score between 0 (least likely) and 1 (most likely) based on some weighted relationship between different shapes.

.. _predictByDistance:

Predict By Distance
===================
calculated a weighted distance based on the distance between 2 or more shapes.

.. autoclass:: prospectpredictor.PredictorByDistance.PredictorByDistance

Prediction
++++++++++

Distance Matrix
----------------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.distanceMatrix

Predictor
---------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.predict

Prediction Architecture
+++++++++++++++++++++++
Functions for updating the prediction architecture and calculating the predictions for a given raster cell

Update Architecture
-------------------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.updateArchitect

Weighted Exp Varriogram
-----------------------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.varriogramExp

Varriogram Power Calculation
----------------------------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.calcPower

Plotting Functions
++++++++++++++++++

Plot Prediction
---------------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.plotPrediction

Plot distance Matrix
--------------------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.plotDistance

Saving/Loading prediction state
+++++++++++++++++++++++++++++++

Save current state
------------------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.saveRaster

Load state
----------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.loadRaster

Utilities
++++++++++

return corrdinate
-----------------
.. automethod:: prospectpredictor.PredictorByDistance.PredictorByDistance.xy






