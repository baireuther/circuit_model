# Copyright 2017 Paul Baireuther. All Rights Reserved.
#
# ====================================================

"""
This code implements a simplified Pauli error channel model for odd distance,
square shaped surface code with rough edges [1]. The error model follows the
circuit model described in [2] with some deviations which are discussed in [3].
In particular it does not include correlated two-qubit errors during the CNOT
gates. To avoid hook errors it uses the improvements suggested in [4]. The code
layout and circuit are for example illustrated in figure 1 of [3].

References
----------
[1] H. Bombin and M. A. Martin-Delgado, Phys. Rev. A 76, 012305 (2007)
[2] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland,
    Phys. Rev. A 86, 032324 (2012)
[3] P. Baireuther, T. E. O'Brien, B. Tarasinski, and C. W. J. Beenakker,
    Quantum 2, 48 (2018)
[4] Y. Tomita and K. M. Svore, Phys. Rev. A 90, 062320 (2014)
"""

import sqlite3
import numpy as np
import copy


class SurfaceCode:
  """ This class describes a square surface code with an odd distance 3,5,7,...

  Input
  -----

  seed -- a seed to initialize the random number generator
  git_version -- git version of this file, if it is not available the default
                 is set to zero
  distance -- the distance of the surface code
  pqx, pqy, pqz -- error rates on the data qubits (per circuit element)
  pax, pay, paz -- error rates on the ancilla qubits (per circuit element)
  pm -- measurement errors applied at both ancilla and data qubit readouts
  """

  def __init__(self, seed, git_version=0, distance=3,
               pqx=0, pqy=0, pqz=0, pax=0, pay=0, paz=0, pm=0):

    # # # Git version and seed # # #
    self.git_version = git_version
    self.seed = seed

    # # # Create an instance of random # # #
    self.rng = np.random.RandomState(self.seed)

    # # # Set variables # # #
    self.dist = distance
    self.n_data = self.dist * self.dist  # number of data qubits
    self.n_anc = self.n_data - 1  # number of ancilla (anc) qubits
    self.n_z_stab = int(self.n_anc / 2)  # number of z-stabilizers
    self.pqx, self.pqy, self.pqz = pqx, pqy, pqz
    self.pax, self.pay, self.paz = pax, pay, paz
    self.pm = pm

    # # # Initialize data and ancilla qubits # # #
    self._init_qubits()

    # # # Initialize CNOT gates # # #
    self._init_cnots()

  def _init_qubits(self):
    """ This function initializes both ancilla and data qubits. It also
    creates lists with the positions of all data, x-ancilla, and z-ancilla
    qubits.

    Each qubit is defined by two booleans (bitflip-error, phaseflip-error),
    i.e. 00 - no error, 10 - x-error, 11 - y-error, 01 - z-error.
    """

    # Initialize arrays to hold the error information. The qubits will be
    # arranged in the geometry of the surface code.
    self.data_qubits = np.zeros(
        shape=[self.dist, self.dist, 2], dtype=bool)
    self.anc_qubits = np.zeros(
        shape=[self.dist + 1, self.dist + 1, 2], dtype=bool)

    # Generate lists with qubit positions.
    self.data_l = [(m, n) for m in range(self.dist) for n in range(self.dist)]
    self.x_anc_l, self.z_anc_l = [], []
    for m in range(self.dist + 1):
      for n in range(self.dist + 1):
        if self._anc_exists(m, n):
          if np.mod(m + n, 2) == 0:
            self.x_anc_l.append((m, n))
          else:
            self.z_anc_l.append((m, n))

    # To produce small outputs, we can condense the matrix of ancilla qubits
    # into a one dimensional vector. Here we do this, and create some meta
    # information about which ancillas in this list correspond to x- and
    # which to z-stabilizer measurements.
    self.anc_l = list(sorted(self.x_anc_l)) + list(sorted(self.z_anc_l))
    self.x_indcs, self.z_indcs = [], []
    for ncond in range(len(self.anc_l)):
      if self.anc_l[ncond] in self.x_anc_l:
        self.x_indcs.append(ncond)
      elif self.anc_l[ncond] in self.z_anc_l:
        self.z_indcs.append(ncond)
      else:
        raise ValueError("ancilla is neither x- nor z")
      ncond += 1

  def _init_cnots(self):
    """ This function builds dictionaries that describe which ancilla
    qubits are coupled to which data qubits by the different CNOT gates.
    These gates are called North, East, South, and West in a cyclic manner
    starting from the top right corner.

    This function also calculates a dictionary with all the connections
    between z-ancillas and data qubits via the 4 CNOT gates. This is later
    used to calculate the final stabilizer from the data qubit measurement.
    """

    # Get and set the CNOT dictionaries
    self.x_north_dict, self.x_east_dict, self.x_south_dict, \
        self.x_west_dict = self._make_cnots(self.x_anc_l)
    self.z_north_dict, self.z_east_dict, self.z_south_dict, \
        self.z_west_dict = self._make_cnots(self.z_anc_l)

    # All connections of z-ancillas to data qubits form a dictionary where the
    # keys are the z-ancilla positions and the entries are lists with all the
    # data qubits that are connected to the corresponding ancilla qubit via
    # the CNOT gates
    self.z_anc_data_conn = {}
    for qb in self.z_anc_l:
      self.z_anc_data_conn[qb] = list()
    for qb in self.z_north_dict.keys():
      self.z_anc_data_conn[qb].append(self.z_north_dict[qb])
    for qb in self.z_east_dict.keys():
      self.z_anc_data_conn[qb].append(self.z_east_dict[qb])
    for qb in self.z_south_dict.keys():
      self.z_anc_data_conn[qb].append(self.z_south_dict[qb])
    for qb in self.z_west_dict.keys():
      self.z_anc_data_conn[qb].append(self.z_west_dict[qb])

  def _reinitialize(self, seed):
    """ This function reinitializes the qubits and sets a new seed.

    Input
    -----
    seed - a new seed for the random number generator
    """

    # Reinitialize random number generator
    self.seed = seed
    self.rng = np.random.RandomState(seed)

    # Reinitialize qubits
    self.data_qubits = np.zeros(
        shape=[self.dist, self.dist, 2], dtype=bool)
    self.anc_qubits = np.zeros(
        shape=[self.dist + 1, self.dist + 1, 2], dtype=bool)

  def _make_cnots(self, anc_l):
    """ This function, given a list of ancilla qubit positions, returns
    four dictionaries which describe how the ancilla qubits are connected
    to the data qubits via the four CNOT gates.

    Input
    -----
    anc_l -- list of ancilla qubit positions

    Output
    ------
    north_dict, east_dict, south_dict, west_dict -- four dictionaries,
    that describe how the ancillas are connected to data qubits for a
    given CNOT gate in the order north, east, south, west
    """

    north_dict, east_dict, south_dict, west_dict = {}, {}, {}, {}
    for m, n in anc_l:
      if m > 0 and n < self.dist:
        north_dict[(m, n)] = (m - 1, n)
      if m < self.dist and n < self.dist:
        east_dict[(m, n)] = (m, n)
      if m < self.dist and n > 0:
        south_dict[(m, n)] = (m, n - 1)
      if m > 0 and n > 0:
        west_dict[(m, n)] = (m - 1, n - 1)
    return north_dict, east_dict, south_dict, west_dict

  def _do_step_1(self):
    """ This function executes the first step of the circuit model. During
    this step the x-ancillas undergo a Hadamard rotation, the z-ancillas
    idle. Both, data- and ancilla qubits experience uncorrelated errors.
    """

    # Apply a Hadamard gate on the x-ancillas
    self._hadamard_on_x_ancs()

    # Apply uncorrelated errors to all qubits
    self._apply_uncorr_errs(self.x_anc_l + self.z_anc_l, 'anc')
    self._apply_uncorr_errs(self.data_l, 'data')

  def _do_step_2(self):
    """ This function executes the second step of the circuit model. During
    this step North CNOT gates are applied to the x-ancillas, and North
    CNOT gates are applied to the z-ancillas.
    """
    self._do_cnot_step(self.x_north_dict, self.z_north_dict)

  def _do_step_3(self):
    """ This function executes the third step of the circuit model. During
    this step West CNOT gates are applied to the x-ancillas, and East CNOT
    gates are applied to the z-ancillas.
    """
    self._do_cnot_step(self.x_west_dict, self.z_east_dict)

  def _do_step_4(self):
    """ This function executes the fourth step of the circuit model. During
    this step East CNOT gates are applied to the x-ancillas, and West CNOT
    gates are applied to the z-ancillas.
    """
    self._do_cnot_step(self.x_east_dict, self.z_west_dict)

  def _do_step_5(self):
    """ This function executes the fifth step of the circuit model. During
    this step South CNOT gates are applied to the x-ancillas, and South
    CNOT gates are applied to the z-ancillas.
    """
    self._do_cnot_step(self.x_south_dict, self.z_south_dict)

  def _do_step_6(self):
    """ This function executes the sixth step of the circuit model. It is the
    same as the first step. """
    self._do_step_1()

  def _do_measure_step(self):
    """ This function executes the seventh step. During this step the ancilla
    qubits are are measured (with measurement errors). The measurement
    collapses the wave functions of the ancilla qubits, and the information
    about phaseflip errors is lost (reset). The data qubits idle and
    experience uncorrelated errors.

    Output
    ------
    stabs -- measurement of the stabilizers (=ancilla qubits, =syndrome)
    """

    # Measure the ancilla qubits (i.e. the bit flip errors)
    stabs = copy.copy(self.anc_qubits[:, :, 0])
    # Apply measurement errors
    for qb in self.x_anc_l + self.z_anc_l:
      if self.rng.rand() < self.pm:
        stabs[qb] = not stabs[qb]

    # Reset the phase error information
    self.anc_qubits[:, :, 1] = False

    # The data qubits are idling and experience uncorrelated errors
    self._apply_uncorr_errs(self.data_l, 'data')

    return stabs

  def _do_measure_step_condensed(self):
    """ This function does the same as _do_measure_step, but condenses the
    output (stabilizers) to a list without the dummy ancillas. The ordering is
    according to line m and column n.

    Output
    ------
    stabs_condensed -- measurement of the stabilizers in a condensed list
    """
    stabs = self._do_measure_step()
    stabs_condensed = np.array([stabs[qb] for qb in self.anc_l])
    return stabs_condensed

  def _get_parity_of_bitflips(self):
    """ This function returns the parity of the number of x- and y-errors
        (bitflips) on the data qubits.

    Output
    ------
    parity -- parity of the number of x- and y-errors on the data qubits
    """
    n_xy_errs = 0
    for row in self.data_qubits:
      for el in row:
        if el[0]:
          n_xy_errs += 1
    parity = bool(np.mod(n_xy_errs, 2))
    return parity

  def _calc_final_z_stabs(self):
    """ This function calculates the final z-stabilizers from the measured
    data-qubits. During the measurement of the data qubits errors occur.
    These errors need to be taken into account when calculating the parity
    of bitflips. The parity of the measurement errors is also returned by
    this function.

    Output
    ------
    z_stabs -- final z-stabilizers
    meas_parity -- parity of measurement errors
    """

    # Measure data qubits (loose phase information)
    data_qubits_meas = copy.copy(self.data_qubits[:, :, 0])
    # Apply measurement errors
    d1, d2 = np.shape(data_qubits_meas)
    m_errs = self.rng.rand(d1, d2) < self.pm
    data_qubits_meas = np.bitwise_xor(data_qubits_meas, m_errs)
    meas_parity = np.mod(sum(sum(m_errs)), 2)

    # Calculate final stabilizers. The idea is to start with clean
    # ancilla qubits and then flip them for each measured bitflip
    # error on the neighboring data qubits.
    z_stabs = np.zeros(shape=[self.dist + 1, self.dist + 1], dtype=bool)
    for anc_qb in self.z_anc_data_conn.keys():
      data_qb_l = self.z_anc_data_conn[anc_qb]
      for data_qb in data_qb_l:
        if data_qubits_meas[data_qb]:
          z_stabs[anc_qb] = not z_stabs[anc_qb]

    return z_stabs, meas_parity

  def _calc_final_z_stabs_condensed(self):
    """ This function does the same as _calc_final_z_stabs, but
    condenses the output (stabilizers) to a list containing only the
    z-ancillas. The ordering is according to line m and column n.

    Output
    ------
    z_stabs -- final z-stabilizers in a list
    meas_parity -- parity of measurement errors
    """

    z_stabs, meas_parity = self._calc_final_z_stabs()
    z_stabs = np.array([z_stabs[qb] for qb in sorted(self.z_anc_l)])
    return z_stabs, meas_parity

  def _do_cnot_step(self, x_dict, z_dict):
    """ This function executes one of the CNOT steps. It applies the CNOT
    operations for both ancilla and data qubits.

    Input
    -----
    x-dict -- a dictionary containing the connections set by the CNOT
              gates between x-ancillas and data qubits
    z-dict -- like x-dict, but for z-ancillas """

    # Apply CNOTS
    self._do_cnots(x_dict, 'x')
    self._do_cnots(z_dict, 'z')

    # Apply uncorrelated errors to all qubits
    self._apply_uncorr_errs(self.x_anc_l + self.z_anc_l, 'anc')
    self._apply_uncorr_errs(self.data_l, 'data')

  def _apply_uncorr_errs(self, qubits, which_qubits):
    """ This function applies uncorrelated errors to the qubits.

    Input
    -----
    qubits -- a list with all the qubits that are subject to uncorrelated
              errors (must be either all data- or all ancilla-qubits)
    which_qubits -- a string describing the type of qubits in 'qubits':
                    'data' for data qubits, 'anc' for ancilla qubits
    """

    if which_qubits == 'data':
      for qb in qubits:
        # We throw the dice three times for independent x-, y- and z-errors
        rnx = self.rng.rand()
        rny = self.rng.rand()
        rnz = self.rng.rand()
        if rnx < self.pqx:
          self.data_qubits[qb][0] = not self.data_qubits[qb][0]
        if rny < self.pqy:
          self.data_qubits[qb][0] = not self.data_qubits[qb][0]
          self.data_qubits[qb][1] = not self.data_qubits[qb][1]
        if rnz < self.pqz:
          self.data_qubits[qb][1] = not self.data_qubits[qb][1]

    elif which_qubits == 'anc':
      # We throw the dice three times for independent x-, y- and z-errors
      for qb in qubits:
        rnx = self.rng.rand()
        rny = self.rng.rand()
        rnz = self.rng.rand()
        if rnx < self.pax:
          self.anc_qubits[qb][0] = not self.anc_qubits[qb][0]
        if rny < self.pay:
          self.anc_qubits[qb][0] = not self.anc_qubits[qb][0]
          self.anc_qubits[qb][1] = not self.anc_qubits[qb][1]
        if rnz < self.paz:
          self.anc_qubits[qb][1] = not self.anc_qubits[qb][1]

    else:
      raise ValueError("which_qubits must be 'data' or 'anc' but is " +
                       str(which_qubits))

  def _hadamard_on_x_ancs(self):  # test written
    """ This function applies a Hadamard gate to the x-ancillas. This gate
    exchanges bitflip- and phaseflip-errors, i.e. x <--> z errors.
    Y errors get a global phase that we can ignore.
    """
    for qb in self.x_anc_l:
      e1, e2 = self.anc_qubits[qb]
      self.anc_qubits[qb] = [e2, e1]

  def _do_cnots(self, cnots, which_anc):
    """ This function executes the CNOT gates.
    For x-ancillas: ancilla qubits are control bits, data qubits are
                    target bits
    For z-ancillas: ancilla qubits are target bits, data qubits are
                    control bits

    Input
    -----
    cnots -- a dictionary that describes how the cnots are connecting
             ancillas and data qubits
    which_anc -- a string, saying whether the CNOTs connect to
                 x- or z-ancillas
    """

    # Figure out what needs to be done
    for anc_qb in cnots.keys():
      data_qb = cnots[anc_qb]
      anc_err = self.anc_qubits[anc_qb]
      data_err = self.data_qubits[data_qb]
      if which_anc == 'x':
        anc_err, data_err = self._get_cnot_action(anc_err, data_err)
      elif which_anc == 'z':
        data_err, anc_err = self._get_cnot_action(data_err, anc_err)
      else:
        raise ValueError("which_anc must be 'x' or 'z', but is " +
                         str(which_anc))

      # Write the results onto the qubits
      self.anc_qubits[anc_qb] = anc_err
      self.data_qubits[data_qb] = data_err

  def _get_cnot_action(self, c, t):  # test written
    """ This function describes the action of the CNOT gates.

    Input
    -----
    c -- errors on control qubit
    t -- errors on target qubit

    Output
    -----
    c -- errors on control qubit
    t -- errors on target qubit
    """

    # X- or y-error on control
    if c[0]:
      t[0] = not t[0]

    # Y- or z-error on target
    if t[1]:
      c[1] = not c[1]

    return c, t

  def _anc_exists(self, m, n):
    """ This function checks if an ancilla qubit exist or is a dummy """
    fake_ancillas = [(m == 0 and n == 0),
                     (m == self.dist and n == self.dist),
                     (m == 0 and np.mod(n, 2) == 1),
                     (m == self.dist and np.mod(n, 2) == 0),
                     (np.mod(m, 2) == 0 and n == 0),
                     (np.mod(m, 2) == 1 and n == self.dist)
                     ]
    if any(fake_ancillas):
      return False
    else:
      return True

  def make_run(self, seed, n_steps, condensed=True):
    """ This function first reinitializes the system, and the calculates
    a ('measurement') n_step steps. Note that since we return a final
    stabilizer at each step, it corresponds to n_step actual measurements
    which have the same syndromes (up to the given step where the data
    qubits are read out).

    Input
    -----
    seed -- a seed for the random number generator
    n_steps -- the number of steps (in sets of 7 circuit steps)
    condensed -- a flag determining if the output should be in condensed
                 lists or in arrays that resemble the geometry of the
                 surface code

    Output
    ------
    seed -- the seed that was used for the run
    syndromes -- syndromes
    events -- second derivatives of syndromes
    fstabs -- the final stabilizer calculated from data qubit measurements
    err_signal -- the error signal, which is xor(fstabs, first_deriv)
    parities -- the parities of the bitflip errors on the data qubits
                (combined x- and y-errors)
    """

    # Reinitialize the system
    self._reinitialize(seed)

    # Execute the seven substeps n_step times
    syndromes, fstabs, parities = [], [], []
    for s in range(n_steps):
      self._do_step_1()
      self._do_step_2()
      self._do_step_3()
      self._do_step_4()
      self._do_step_5()
      self._do_step_6()

      # The final step is a bit tricky because we do both the 7th step and the
      # final measurement simultaneously.

      # FIRST we must do the final measurement, otherwise we add extra errors.
      if condensed:
        z_fstabs, parity_meas = self._calc_final_z_stabs_condensed()
      else:
        z_fstabs, parity_meas = self._calc_final_z_stabs()
      fstabs.append(z_fstabs)
      # The final parity is the parity of the bitflips that occurred on the
      # data qubits + the number of bit flips that occur during the
      # measurement of the data qubits ('final measurements').
      parity_clean = self._get_parity_of_bitflips()
      final_parity = parity_clean != parity_meas
      parities.append(final_parity)

      # SECOND we do the step 7
      if condensed:
        syndromes.append(self._do_measure_step_condensed())
      else:
        syndromes.append(self._do_measure_step())

    syndromes = np.array(syndromes)
    fstabs = np.array(fstabs)
    parities = np.array(parities)

    # Finally, we calculate the first and second derivative (events)
    # of the syndromes and the final error signal.
    first_deriv = []
    for s in range(n_steps):
      if s < 1:
        first_deriv.append(syndromes[s])
      else:
        first_deriv.append(np.bitwise_xor(
            syndromes[s], syndromes[s - 1]))
    first_deriv = np.array(first_deriv, dtype=bool)

    second_deriv = []
    for s in range(n_steps):
      if s < 2:
        second_deriv.append(syndromes[s])
      else:
        second_deriv.append(np.bitwise_xor(
            syndromes[s], syndromes[s - 2]))
    events = np.array(second_deriv, dtype=bool)

    err_signal = []
    if condensed:
      for s in range(n_steps):
        err_signal.append(np.bitwise_xor(
            fstabs[s], first_deriv[s][self.z_indcs]))
    else:
      for s in range(n_steps):
        first_deriv_z_only = np.zeros(
            shape=[self.dist + 1, self.dist + 1], dtype=bool)
        for z_anc in self.z_anc_l:
          first_deriv_z_only[z_anc] = first_deriv[s][z_anc]
        err_signal.append(np.bitwise_xor(
            fstabs[s], first_deriv_z_only))
    err_signal = np.array(err_signal, dtype=bool)

    return seed, syndromes, events, fstabs, err_signal, parities

  def get_info(self):
    """ This function returns some information about the variables that
    describe the surface code model.
    """
    return {'git_version': self.git_version, 'seed': self.seed,
            'distance': self.dist,
            'pqx': self.pqx, 'pqy': self.pqy, 'pqz': self.pqz,
            'pax': self.pax, 'pay': self.pay, 'paz': self.paz,
            'pm': self.pm,
            'n_data_qubits': self.n_data, 'n_anc_qubits': self.n_anc,
            'n_z_stabs': self.n_z_stab}


def convert_simple(data, Nmin, Nmax):
    # The circuit model outputs a final syndrome increment and a parity after
    # each error correction cycle. This function removes all of them except the
    # one after the last error correction cycle. The number of cycles iterates
    # between Nmin and Nmax.
  n = Nmin
  data_converted = []
  for dat in data:
    seed, syndromes, events, fstabs, err_signals, parities = dat
    d1, d2 = np.shape(events)

    # In the version used in [3] the network requires input vectors of
    # equal length, we therefore buffer the error cycles with zeros up
    # to the n_steps_max.
    event = np.concatenate(
        (events[:n], np.zeros((Nmax - n, d2), dtype=bool)), axis=0)
    err_sig = err_signals[n - 1]
    parity = parities[n - 1]
    length = n

    # # # # # # # # # # # # # # NOTE  # # # # # # # # # # # # # # # # #
    # It does not make much sense to put the seed and the length in   #
    # np.arrays. This is only kept in order to reproduce the original #
    # data sets used in [3].                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    data_converted.append((np.array([seed]), event, err_sig, parity,
                           np.array([length])))

    # Reset cycle number n if Nmax has been reached.
    if n == Nmax:
      n = Nmin
    else:
      n += 1

  return data_converted


if __name__ == '__main__':

  # # # GIT VERSION # # #
  # If the error model is not under git version control,
  # this variable can be set to zero.
  error_model_gitv = 0

  # # # CODE DISTANCE # # #
  dist = 3

  # # # MODE # # #
  # Training data is used for training, validation data for feedback
  # during training and early stopping, and test data for the curves
  # shown in the paper [3].
  # mode = 0: training data set, seeds from 0 ... N_train
  # mode = 1: validation data set, seeds from 10**8 ... 10**8 + N_validation
  # mode = 2: testing data set, seeds from 2*10**8 ... 2*10**8 + N_test
  mode = 0

  """ WARNING: Existing databases will be overwritten! """
  # Directory where the database will be stored
  db_path =

  # The file name will be base + suffix
  base = "surf17"

  # The suffix is generate according to the mode.
  if mode == 0:
    suffix = "_train.db"
  elif mode == 1:
    suffix = "_validation.db"
  elif mode == 2:
    suffix = "_test.db"

  fname = base + suffix
  print("The database will be written to", db_path + fname + suffix)

  # # # PARAMETERS OF THE ERROR MODEL # # #

  # (Approximate) physical error rate per cycle, assuming px=py=pz.
  p_phys = 0.01

  # In figure 4 of [3] we increase the y-error rate using a prefactor fy,
  # from fy=0 to fy=2. fy=1 corresponds to an isotropic error model.
  fy = 1

  # # # AUTOMATICALLY CALCULATED PARAMETERS # # #

  # There are seven steps in the circuit model, the error probability
  # per step is given by
  p_per_step = p_phys / 7.

  # There are x, y, and z-errors. The error probability per qubit and
  # step is set to
  p = p_per_step / 3.

  # X-, y-, z-error probabilities on the data qubits.
  pqx, pqy, pqz = p, p * fy, p

  # X-, y-, z-error probabilities on the ancilla qubits.
  pax, pay, paz = p, p * fy, p

  # Measurement error probability at readout (same for ancilla- and
  # data-qubits).
  pm = p_per_step

  # # # DETAILS REGARDING THE DIFFERENT DATA SETS # # #

  # The seeds are N0, N0+1, ..., N0+N_samples-1.
  N0 = mode * 10**8

  # Each data set contains N_samples examples, which consists of n_steps_min to
  # n_steps_max error cycles.
  if mode == 0:
    N_samples = 4 * 10**6
    n_steps_min, n_steps_max = 11, 20
  elif mode == 1:
    N_samples = 10**4
    n_steps_min, n_steps_max = 81, 100
  elif mode == 2:
    N_samples = 5 * 10**4
    n_steps_min, n_steps_max = 1, 500

  # Generate seeds.
  seeds = range(N0, N0 + N_samples)

  print("Error probability on the physical data qubits in percent: (x, y, z) =",
        round(pqx * 100, 4), round(pqy * 100, 4), round(pqz * 100, 4))
  print("Error probability on the ancilla qubits in percent: (x, y, z) =",
        round(pax * 100, 4), round(pay * 100, 4), round(paz * 100, 4))
  print("Measurement error probability on both ancilla and data qubits in " +
        "percent:", round(pm * 100, 4))

  # # # DATABASE # # #

  # Generate the database
  conn = sqlite3.connect(db_path + fname)
  c = conn.cursor()

  # Create tables
  c.execute('''DROP TABLE IF EXISTS data''')
  c.execute('''DROP TABLE IF EXISTS info''')
  conn.commit()

  # Table with info about the error rates
  c.execute('''CREATE TABLE info (error_model_gitv, distance, pqx, pqy, pqz, pax,
             pay, paz, pm)''')
  entries = [(error_model_gitv, dist, pqx, pqy, pqz, pax, pay, paz, pm)]
  c.executemany('INSERT INTO info VALUES (?,?,?, ?,?,?, ?,?,?)', entries)

  if mode == 0 or mode == 1:
    # table for the data
    c.execute('''CREATE TABLE data (seed, events, err_signal, parity,
                 length)''')
    # seed is unique index
    c.execute('''CREATE UNIQUE INDEX idx_data_seed ON data(seed)''')
  elif mode == 2:
    # table for the data
    c.execute('''CREATE TABLE data (seed, syndromes, events, fstabs,
                 err_signal, parities)''')
    # seed is unique index
    c.execute('''CREATE UNIQUE INDEX idx_data_seed ON data(seed)''')

  conn.commit()

  # # # GENERATE AN INSTANCE OF THE CIRCUIT MODEL # # #
  surf = SurfaceCode(seed=0, git_version=error_model_gitv, distance=dist,
                     pqx=pqx, pqy=pqy, pqz=pqz,
                     pax=pax, pay=pay, paz=paz,
                     pm=pm)

  # # # TRAINING AND VALIDATION DATA # # #

  # This is data that is used by the network during training (directly or
  # indirectly). Since we want to claim that the network can be trained on
  # experimentally accessible data we can only use a single final stabilizer
  # measurement and parity from each run.

  if mode == 0 or mode == 1:
    # We evaluate the error circuit
    runs = []
    for seed in seeds:
      runs.append(surf.make_run(seed=seed, n_steps=n_steps_max,
                                condensed=True))

    # We remove all data that could not be obtained in an experiment and also
    # data that we do not need in order to to save memory (for example
    # syndromes and error signals contain the same information, and the
    # network uses only the error signals.
    runs_processed = convert_simple(runs, Nmin=n_steps_min, Nmax=n_steps_max)

    # save in database
    c.executemany('REPLACE INTO data VALUES (?, ?, ?, ?, ?)', runs_processed)
    conn.commit()
    conn.close()

  # # # TESTING DATA # # #

  # During testing we "oversample" the output of the error model, i.e., we
  # store the final error signal and  parity after every stabilizer measurement
  # cycle.
  if mode == 2:
    # evaluate the error circuit
    runs = []
    for seed in seeds:
      runs.append(surf.make_run(seed=seed, n_steps=n_steps_max,
                                condensed=True))

    # save in database
    c.executemany('REPLACE INTO data VALUES (?, ?, ?, ?, ?, ?)', runs)
    conn.commit()
    conn.close()

  print("DONE")
