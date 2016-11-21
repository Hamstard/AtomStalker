# AtomStalker

AtomStalker is a module which allows the user to customize shapes of atoms they are interest in tacking in scanning transmission electron miscrocope images. Currently this module allows to easily generate suggestions for positive and negative samples training an scikit-learn classifier (assuming the atoms of interest are the brightest objects). This approach allows the user to customize what is supposed to be identified without the assumption of an intensity model. Once the classifier is trained to the users satisfaction this module can also be used to generate trajectories.
