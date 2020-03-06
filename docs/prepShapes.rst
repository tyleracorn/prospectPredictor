.. _prepShapes:
.. public

Prepping Shapefiles
###################

The prepShapes class included a general wrapper for taking GIS shapefiles and grouping shapes into groups of interest. For instance you can select all polygons under the "rock_class" category that mach the description "ultramafic rocks", and set that as one group. Then select all polygons under the "rock_type" category that match the description "granodioritic intrusive rocks" and set that as your second group.

The prepShapes class also includes usefull functions such as creating a buffer around the shapes and using that to find a smaller project boundary extents.

prepShapes Class 
================
.. autoclass:: prospectpredictor.prepShapes.PrepShapes

Set instance attributes
+++++++++++++++++++++++

Set shapes of interest
----------------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.setShapes

Set buffer size
---------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.setBuffer

Set project boundary
--------------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.setProjectBoundary

Exploring the shapefiles
++++++++++++++++++++++++
Functions for exploring the shape files

print Columns
-------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.printColumns

print unique values in a column
-------------------------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.printUnique

plot shapes grouped within column
----------------------------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.plotData

plot shapes grouped by interest
-------------------------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.plotShapes

Utilities functions
+++++++++++++++++++
Functions to apply to the shape files

dissolve polygons
-----------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.dissolveData

buffer polygons
---------------
.. automethod:: prospectpredictor.prepShapes.PrepShapes.bufferData