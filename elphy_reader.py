# -*- coding: utf-8 -*-
"""
Module elphy_reader.py
----------------------

This module provides a class ElphyFile that allows accessing the data inside a .DAT file saved by Elphy.

Usage: 
    x = ElphyFile(filename, read_data=false) # by default the data is not read but only the objects, use read_data=true to also read the data
    
The returned ElphyFile object has the following properties:
    - file_name, file_size   # file info
    - initial_objects                 # array of objects that appear before
    the first episode in the file
    - n_episodes                         # number of episodes (=1 for continuous recording)
    - episodes                             # array of ElphyEpisode objects
    
An ElphyEpisode object has the following properties (among others):
    - nchan                                 # number of acquired channels. Include the tag channel if any
    - nbpt                                    # nominal number of samples per channel (i.e. before any subsampling)
    - uX                                        # time units (generally 'ms' or 'sec' )
    - x0u, dxu                            # time offset and sampling period (in uX units)
    - continuous                        # indicates whether acquisition is performed in Continuous mode
    - channels                            # array of ElphyChannel objects
    - objects                # array of objects that appear within this episode
    
An ElphyChannel object has the following properties:
    - uX                                        # time units (generally 'ms' or 'sec' )
    - x0u, dxu                            # time offset and sampling period (in uX units)
                                                    # sampling period differ from the parent episode sampling period if there is any subsampling
    - uY                                        # vertical units
    - Ktype                                 # data type
    - data                                    # the data
    
Details of the Elphy format can be found in the file ELPHYFORMAT.HTML in the same folder.
    
"""

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# IMPORT

import numpy as np
import struct
import os
import io
import re
from datetime import timedelta, datetime
from pprint import pprint
from collections import OrderedDict

# typing is available only in Python 3
import sys
import collections

if sys.version_info >= (3,):
    from typing import List, BinaryIO, Any

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# UTILS

# symbols of types that could
# encode a value in an elphy file
elphy_types = (
    'B',
    'b',
    'h',
    'H',
    'l',
    'f',
    'real48',
    'd',
    'ext',
    's_complex',
    'd_complex',
    'complex',
    'none'
)


def disp(x):
    if hasattr(x, '__dict__'):
        x = vars(x)
    pprint(x)


def _read_extended(file_handle):
    x = fread(file_handle, 'Q') * 2. ** -63
    s_e = fread(file_handle, 'h')
    s = np.sign(s_e)
    e = abs(s_e) - (2 ** 14 - 1)
    try:
      return np.sign(s_e) * x * 2. ** e
    except:
      return np.nan


def fread(file_handle, bin_type, n=None):
    # type: (BinaryIO, str, int) -> Any
    """
    file_handle = buffered reader
    bin_type = 'b'=binary, 's'=string or 'h'=, etc. (see struct module)
               or 'str' for string
    n = number of elements to be read, if specified, a numpy array is
        returned (or a string if bin_type is 'str'), otherwise a single
        numerical value is returned if bin_type contains specification for a
        single value, or a tuple for multiple values
    """
    # size in bytes of the coding
    # bintype: b = signed Char, s = char, h = short integer

    # special: extended
    if bin_type == 'ext':
        val = [_read_extended(file_handle) for i in range(n)]
        if n == 1:
            val = val[0]
        else:
            val = np.array(val)
        return val

    # special: string
    do_str = (bin_type == 'str')
    if do_str:
        bin_type = str(n) + 's'
        n = 1

    s = struct.calcsize(bin_type)
    if n is not None:
        s *= n
    data = file_handle.read(s)
    if do_str:
        val = data.decode()
    elif n is None:
        val = struct.unpack(bin_type, data)  # returns a tuple: bin_type can contain multiple type indications
        if len(val) == 1:
            val = val[0]
    else:
        m = re.search('[<>]', bin_type)  # indication of bytes order?
        if m:
            m = m.group(0)
            bin_type = bin_type.replace(m, '').replace(' ', '')
            bin_type = np.dtype(bin_type)
            bin_type = bin_type.newbyteorder(m)
        val = np.frombuffer(data, dtype=bin_type)
    return val


def _read_string(file_handle, max_size=None, size_bin_type='B',
                 return_size=False):
    # read a Pascal string, returns a byte array
    size = fread(file_handle, size_bin_type)
    if max_size:
        size = np.minimum(size, max_size)
    x = fread(file_handle, 'str', size)

    if max_size:
        file_handle.seek(max_size - size, 1)
    if return_size:
        return x, struct.calcsize('B') + size
    else:
        return x


def _sub_block_ID(f):
    ID = _read_string(f)
    SZ = fread(f, '< H')
    if SZ == 0xFFFF:  # then is larger than this, new version
        SZ = fread(f, 'I')
    pos_end = f.tell() + SZ
    return ID, SZ, pos_end


def _decode_user_info(info, key):
    if isinstance(info, dict):
        # already decoded
        return info

    stream = io.BytesIO(info)
    typ_map = {'bool': 'B', 'int': '< i', 'ext': 'ext'}
    decoded = OrderedDict()
    for name, typ in key.items():
        if typ == 'str':
            value = _read_string(stream)
        else:
            shape = None
            if isinstance(typ, tuple):
                n = typ[1]
                if isinstance(n, tuple):
                    shape = n
                    n = int(np.array(shape).prod())
                typ = typ[0]
            else:
                n = 1
            if typ == 'str':
                value = _read_string(stream, max_size=n)
            else:
                value = fread(stream, typ_map[typ], n)
            if shape:
                value = value.reshape(shape)
            if typ == 'bool':
                if n == 1:
                    value = bool(value)
                else:
                    value = value.astype(bool)
        decoded[name] = value

    return decoded


def _gcd(x, y):
    # Greatest common divider
    while y:
        x, y = y, x % y
    return x


def _lcm(a, b):
    # Lowest common multiple
    return abs(a * b) // _gcd(a, b) if a and b else 0


def _GetChanBytesMask(KS, Kbytes):
    # Compute the "mask" of how different channels are saved after downsampling
    # KS is the down-sampling factor of each channel
    # Kbytes is the number of bytes for each channel
    # The mask will describe to each channel belongs each successive byte.
    # For example, if ther are 4 channels, channel 2 is converted into events
    # (i.e. no data is recorded) and channel 4 is downsampled by a factor 5,
    # then the "channel mask" is 1 3 4 1 3 1 3 1 3 1 3.
    # If channel 2 is saved over 4 bytes, 3 over 1 byte and 4 over 4 bytes,
    # then the "bytes channel mask" is 1 1 3 4 4 4 4 1 1 3 1 1 3 1 1 3 1 1 3.

    Nvoie = len(KS)

    # Number of sample acquisitions inside a mask
    Klcm = 1
    for Ksamp in KS:
        if Ksamp > 0:
            Klcm = _lcm(Klcm, Ksamp)

    # Build the mask
    bytesMask = []
    for iacq in range(Klcm):
        for ch in range(Nvoie):
            Ksamp = KS[ch]
            if Ksamp > 0 and iacq % Ksamp == 0:
                bytesMask += [ch] * Kbytes[ch]
    return np.array(
        bytesMask)  # we need a Numpy Array for indexing by [bytesMask==i]


def _read_rdata_header(file_handle):
    header_size, is_first, PCtime, _ = fread(file_handle, '< H B Q I')
    x = (PCtime * 2 ** -37) - 33286456
    date = datetime.fromordinal(int(x)) + timedelta(
        days=x % 1) - timedelta(days=366)
    return header_size, date


# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# ELPHYFILE CLASS
#
# contains the main loop for reading the file, but reading of specific objects 
# occurs in specialized objects
class ElphyFile(object):
    def __init__(self, filename, read_data=False, read_spikes=None):
        # type: (str, bool, bool) -> None

        # File info
        self.file_name = filename
        self.file_size = None

        # Data
        self.initial_objects = []
        self.file_info = None
        self.n_episodes = 0
        self.episodes = []  # type: List[ElphyFile.Episode]

        # Read
        if read_spikes is None:
            read_spikes = read_data
        self._read(read_data, read_spikes)

    def _read(self, read_data, read_spikes):
        # Open file
        try:
            file_handle = open(self.file_name, 'rb')
        except Exception as e:
            raise Exception(
                "python couldn't open file %s : %s" % (self.file_name, e))
        statinfo = os.stat(self.file_name)
        self.file_size = statinfo.st_size

        # Some info to be displayed after reading
        self._ignoredblocks = []

        # Skip header
        offset = self._skip_header(file_handle)

        # Cycle over all blocks
        while offset < self.file_size:
            offset = self._read_block(file_handle, read_data, read_spikes)

        # Display ignored block identities and incorrect nsamp
        if self._ignoredblocks:
            msg = "All blocks of type '" + "', '".join(
                self._ignoredblocks) + "' were ignored while reading Elphy file"
            print(msg)

        # Close file
        file_handle.close()

    def _skip_header(self, file_handle):
        """
        An Elphy object file begins with a 18-byte header with a fixed structure:
        - File Title : 'DAC2 objects'
        - size : size of the header (always 18)

        Remember that the first byte in a pascal string is the effective length.
            """
        # start of the file
        file_handle.seek(0)
        # return 18 fist bytes, and extract lenght, title and ...
        length, title, dummy = fread(file_handle, '< B 15s H')
        title = title[0:length].decode()
        if not title.startswith('DAC2 objects'):
            raise Exception(
                "only recent Elphy files are accepted ('DAC2 objects'). Try resaving it with Elphy.")
        return file_handle.tell()

    def _read_block(self, file_handle, read_data, read_spikes):
        """
        After the header, we find one or more blocks with the structure:
        - size:    a 4-byte integer. Total size of block on disk.
        - identity: a pascal string with variable length.
        - data:    an array of size-(len(size)+len(identity)) bytes.

        A reader must:
            - read size
            - read the first byte of identity. This gives the length Len of the Pascal string.
            - read Len bytes. This gives identity
        """

        # Current episode
        if self.n_episodes:
            ep = self.episodes[self.n_episodes - 1]
            objects_list = ep.objects
        else:
            ep = None
            objects_list = self.initial_objects

        # Read size and identifier
        offset = file_handle.tell()
        block_size = fread(file_handle, '< i')
        block_end = offset + block_size
        identity, string_size = _read_string(file_handle, return_size=True)
        block_size_rem = block_size - struct.calcsize('< i') - string_size

        # Read block according to its identifier:

        if identity == 'B_Ep':      # New episode
            self.n_episodes += 1
            ep = self.Episode(self, self.n_episodes, file_handle, block_end)
            self.episodes.append(ep)
        elif identity == 'RDATA':   # Channel data
            ep._read_rdata(file_handle, block_size_rem, read_data)
        elif identity == 'Memo':
            objects_list.append(self.Memo(file_handle, block_end))
        elif identity == 'DBrecord':
            objects_list.append(self.DBrecord(file_handle, block_end))
        elif identity == 'Vector':
            # Note that the 'Vector' block contains only the Vector header
            # information, while its data is in the 'DATA' block that follows
            objects_list.append(self.Vector(file_handle, block_end))
        elif identity == 'DATA':
            if not objects_list:
                raise Exception("Encountered 'DATA' block, but there is no current object!")
            v = objects_list[-1]
            if not isinstance(v, ElphyFile.Vector):
                raise Exception("Encoutered 'DATA' block, but current object is not a Vector")
            v._read_data(file_handle, block_end)
        elif identity == 'B_Finfo':
            _, sub_size, _ = _sub_block_ID(file_handle)
            self.file_info = fread(file_handle, 'B', sub_size)  # Store byte array which can be later decoded using method decode_file_info
        elif identity == 'B_Einfo':
            _, sub_size, _ = _sub_block_ID(file_handle)
            ep.episode_info = fread(file_handle, 'B', sub_size)  # Store byte array which can be later decoded using method decode_episodes_info
        elif identity == 'RSPK':
            ep.spikes._read_spikes(file_handle, block_size_rem, read_spikes)
        elif identity == 'RspkWave':
            ep.spikes._read_waves(file_handle, block_size_rem, read_spikes)
        elif identity == 'RCyberTag':
            ep._read_cyber_tags(file_handle, block_size_rem, read_data)
        elif identity not in self._ignoredblocks:
            self._ignoredblocks.append(identity)

        # Go to block end
        file_handle.seek(block_end)

        return block_end

    def decode_file_info(self, key):
        self.file_info = _decode_user_info(self.file_info, key)

    def decode_episodes_info(self, key):
        for ep in self.episodes:
            ep.decode_episode_info(key)


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ELPHYEPISODE CLASS
    #
    # reads the recording data. Note that the nested ElphyChannel object is only a passive
    # object, i.e. data reading actually occurs in the ElphyEpisode method.
    # The functions gcd, lcm, GetAgSampleCount, GetMask, GetSamplePerChan ...
    class Episode(object):
        def __init__(self, ef, episode_num, file_handle, block_end):
            # type: (ElphyFile, int, BinaryIO, int) -> None
            # Initialize a new episode by reading the Ep block

            # Pointer to parent ElphyFile, and remember episode number
            self.elphy_file = ef
            self.episode_num = episode_num

            # Init lists of channels, events, blocks
            self.channels = []  # type: List[ElphyFile.Channel]

            # Init list of episode objects and user info
            self.objects = []
            self.episode_info = None

            # Date will be read in the RDATA block
            self.date = None

            # Read info from the sub-blocks of the Ep block
            offset = file_handle.tell()
            while offset < block_end:

                # Sub-block Header
                sub_identifier, sub_size, _ = _sub_block_ID(file_handle)
                offset = file_handle.tell()

                # Episode main info
                if sub_identifier == 'Ep':
                    self.nbchan = fread(file_handle, '< B')  # number of acquired channels. Include the tag channel if any
                    self.nbpt = fread(file_handle, '< i')  # nominal number of samples per channel
                    self.tpData = fread(file_handle, '< B')  # Sample type    ( not used. See Ktype )
                    self.uX = _read_string(file_handle, 10)  # Time units (generally 'ms' or 'sec' )
                    self.dxu = fread(file_handle, '< d')  # sampling period (in ux units)
                    self.x0u = fread(file_handle, '< d')  # Time of first sample (0 generally)
                    self.continuous = fread(file_handle, '< ?')  # when false, it's an episode file (uX='ms', there are one or more B_Ep block)
                    self.TagMode = fread(file_handle, '< B')  # 0: not tag channel;
                    # 1: digidata mode : in ADC values, bit0 and bit1 are used to store digital inputs
                    # 2: ITC mode :            one 16-bit channel dedicated to digital inputs
                    # 3: CyberK mode :     only transitions on digital inputs are stored in a separate block
                    self.TagShift = fread(file_handle, '< B')  # when Tagmode=1, indicate of how many bits we must shift the data to get true adc values
                    # generally 4 for digidata 1200, 2 for digidata 1322
                    if sub_size > 36:
                        self.DxuSpk = fread(file_handle, '< d')  # CyberK X-scale parameter for waveforms
                        self.x0uSpk = fread(file_handle, '< d')  # CyberK X-scale parameter for waveforms
                        self.nbSpk = fread(file_handle, '< i')  # CyberK: number of samples in waveforms
                        self.DyuSpk = fread(file_handle, '< d')  # CyberK: Y-scale parameter for waveforms
                        self.y0uSpk = fread(file_handle, '< d')  # CyberK: Y-scale parameter for waveforms
                        self.unitXspk = _read_string(file_handle, 10)  # CyberK units
                        self.unitYSpk = _read_string(file_handle, 10)  # CyberK units
                        self.CyberTime = fread(file_handle, '< d')  # cyberK time in seconds
                        self.PCtime = fread(file_handle, '< I')  # PC time in milliseconds
                        # self.DigEventCh = dict_values[21]   # numéro de la voie Evt utilisant les compteurs NI }
                        # self.DigEventDxu = dict_values[22]  # période échantillonnage correspondante }
                        # self.DxEvent = dict_values[23]      # période échantillonnage des voies evt ordinaires. Par défaut, on avait dxu / nbvoie . }
                        # self.FormatOption = dict_values[24] # ajouté le 26 mars 2011 . =0 jusqu'à cette date.

                # Channel info -> create channels
                elif sub_identifier == 'Adc':
                    self._format_option = 0  # Channels data are interleaved
                    for i in range(0, self.nbchan):
                        ech = ef.Channel(self, i, file_handle, self._format_option)
                        self.channels.append(ech)

                # Down-sampling info
                elif sub_identifier == 'Ksamp':
                    # If Ksampling=1 (default), every sample coming from the channel has been stored in the file.
                    # If Ksampling>1, the channel has been down-sampled, and only one sample in Ksampling has been stored.
                    # if Ksampling=0, no analog data has been stored because data from this channel has been converted in events.
                    for ech in self.channels:
                        ech._read_sampling_info(file_handle)

                # Sample size
                elif sub_identifier == 'Ktype':
                    for ech in self.channels:
                        ech._read_data_type_info(file_handle)

                # All channel info at once -> create channels
                elif sub_identifier == 'Adc2':
                    self._format_option = 1  # Channels data are contiguous
                    for i in range(0, self.nbchan):
                        ech = ef.Channel(self, i, file_handle,
                                         self._format_option)
                        self.channels.append(ech)

                # go to end of subblock
                offset += sub_size
                file_handle.seek(offset)

            # Init spikes and waveforms (will be read in 'RSPK' and 'RspkWave' blocks)
            self.spikes = ef.Spikes(self)

            # Init cyber tags (BlackRock only)
            self.cyber_tags = None

        def _read_rdata(self, file_handle, block_size, read_data):
            """
            Raw Data Block
            --------------
            There are two parts in a RDATA block.
            The first part is this single structure::

                TRdataRecord = record
                    MySize:word;             // size of this record. Currently=15
                    SFirst:boolean;        // true when there was a recording pause before this block
                    Stime:double;      // code the PC time
                    Nindex:longword;   // not used
                end;

            The second part contains the data coming from analog inputs and optionally from digital inputs.
            """

            # header info
            header_size, self.date = _read_rdata_header(file_handle)
            self.data_size = block_size - header_size
            self.data_offset = file_handle.tell()

            # check that data_size is as expected
            if type(self) == ElphyFile.Episode and self._format_option == 1:
                if self.data_size < sum([ch.data_size for ch in self.channels]):
                    raise Exception('Episode has missing data, use ElphyFileMissingData class instead of ElphyFile')

            # data
            if read_data:
                data = self._read_data(file_handle)
                for ech, datai in zip(self.channels, data):
                    ech.data = datai

        def _read_cyber_tags(self, file_handle, block_size, read_data):
            # header info
            header_size, date = _read_rdata_header(file_handle)
            self.cyber_tags_size = block_size - header_size
            self.cyber_tags_offset = file_handle.tell()

            # data
            if read_data:
                self._cyber_tags = self._read_cyber_tags_data(file_handle)

        def get_data(self, keep_in_mem=False, channels=None):
            # Get data only from some channels
            if channels is None:
                channels_idx = range(self.nbchan)
            elif isinstance(channels, str):
                flag = channels.lower()
                if flag == '1khz':
                    channels_idx = [i for i, c in enumerate(self.channels) if c.dxu == 1]
                elif flag == '30khz':
                    channels_idx = [i for i, c in enumerate(self.channels) if c.dxu == 1./30]
                else:
                    raise Exception("cannot interpret channels selection '" +
                                    channels + "'")
            elif isinstance(channels, int):
                channels_idx = [channels]
            else:
                channels_idx = channels
            channels = [self.channels[i] for i in channels_idx]

            # Check whether data was already read and stored
            ok = True
            for ech in channels:
                if ech.data is None:
                    ok = False
                    break
            if ok:
                return [ech.data for ech in channels]

            # If not, read all channels from the file
            file_handle = open(self.elphy_file.file_name, 'rb')
            file_handle.seek(self.data_offset)
            data = self._read_data(file_handle)
            if keep_in_mem:
                for (ech, datai) in zip(self.channels, data):
                    ech.data = datai
            return [data[i] for i in channels_idx]

        def _read_data(self, file_handle):
            data = [None] * self.nbchan

            # Data reading depends on format
            if self._format_option == 0:
                # Data are interleaved, compute channel mask to read the data
                Kbytes = [struct.calcsize(elphy_types[ech.Ktype]) for ech in
                          self.channels]
                KS = [ech.Ksamp for ech in self.channels]
                bytesMask = _GetChanBytesMask(KS, Kbytes)
                nMask = len(bytesMask)

                # step 1: maximal number of entire masks
                nChunk = self.data_size // nMask
                raw = file_handle.read(nChunk * nMask)
                raw = np.frombuffer(raw, dtype='B').reshape((nChunk, nMask))
                for i, ech in enumerate(self.channels):
                    encoded_values = np.frombuffer(
                        np.ravel(raw[:, bytesMask == i]), elphy_types[ech.Ktype])
                    data[i] = encoded_values * ech.dyu + ech.y0u

                # step 2: if needed, a remaining truncated mask
                nremain = self.data_size % nMask
                if nremain > 0:
                    raw = file_handle.read(nremain)
                    truncMask = bytesMask[:nremain]
                    raw = np.frombuffer(raw, dtype='B')
                    for i, ech in enumerate(self.channels):
                        encoded_values = np.frombuffer(raw[truncMask == i],
                                                       elphy_types[ech.Ktype])
                        data[i] = np.concatenate(
                                (data[i], encoded_values * ech.dyu + ech.y0u))

            else:
                # Data are contiguous, read channels one after each other (
                # Channel method)
                for i, ch in enumerate(self.channels):
                    data[i] = ch._read_data(file_handle)

            return data

        def get_cyber_tags(self, keep_in_mem=False):
            # Check whether data was already read and stored
            if self.cyber_tags:
                return self.cyber_tags

            # If not, read from the file
            file_handle = open(self.elphy_file.file_name, 'rb')
            file_handle.seek(self.cyber_tags_offset)
            data = self._read_cyber_tags_data(file_handle)
            if keep_in_mem:
                self.cyber_tags = data
            return data

        def _read_cyber_tags_data(self, file_handle):

            # Tags are read the same way as RDATA (interleaved times and values)
            nMask = 6
            nChunk = self.cyber_tags_size // nMask
            raw = file_handle.read(self.cyber_tags_size)
            raw = np.frombuffer(raw, dtype='B').reshape((nChunk, nMask))
            times = np.frombuffer(np.ravel(raw[:, 0:4]), 'I')
            bits = np.unpackbits(raw[:, 4:6]).reshape((nChunk, 16))
            bits = bits[:, [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]]
            
            nt = self.channels[0].nsamp
            data = np.zeros((nt, 16), dtype=bool)            
            
            times = np.concatenate((times, [nt]))
            for i in range(len(bits)):
                data[times[i]:times[i+1]] = bits[i]

            return data

        def get_spikes(self, keep_in_mem=False, do_spikes=True, do_waves=True):
            return self.spikes.get_data(keep_in_mem, do_spikes, do_waves)

        def decode_episode_info(self, key):
            self.episode_info = _decode_user_info(self.episode_info, key)

    class Channel(object):
        def __init__(self, ep, channel_num, file_handle, format_option):
            # type: (ElphyFile.Episode, int, BinaryIO, int) -> None

            # Pointer to parent episode, and remember channel number
            self.episode = ep
            self.channel_num = channel_num

            # Default no data
            self.data = None  # type: np.ndarray

            if format_option == 0:
                # Read vertical info
                self.uY = _read_string(file_handle, 10)  # vertical units
                self.dyu = fread(file_handle,
                                 '< d')  # scaling parameters when tpdata is an integer type
                self.y0u = fread(file_handle,
                                 '< d')  # for an adc value j, the real value is y = dyu*j + y0u

            elif format_option == 1:
                # Read all info at once
                self.ChanType, self.Ktype = fread(file_handle, 'BB')  # 0=analog / 1=event, sample type
                self.nbyte = struct.calcsize(elphy_types[self.Ktype])
                self.imin, self.imax = fread(file_handle, '< ll')
                self.nsamp = self.imax - self.imin + 1
                self.data_size = self.nsamp * self.nbyte
                self._nsamp = self.nsamp          # this will be corrected in the case of missing data
                self._data_size = self.data_size  # this will be corrected in the case of missing data
                self.uX = _read_string(file_handle, 10)  # horizontal units
                self.dxu, self.x0u = fread(file_handle, '< dd')  # for an adc value j, the real value is x = dxu*j + x0u
                self.uY = _read_string(file_handle, 10)  # vertical units
                self.dyu, self.y0u = fread(file_handle, '< dd')  # for an adc value j, the real value is y = dyu*j + y0u
                self.nmissing = None  # will be set in the episode _read_rdata method
                self.data = None

        def _read_sampling_info(self, file_handle):
            self.Ksamp = fread(file_handle, '< H')
            # now we can set the time properties of the channel
            self.uX = self.episode.uX
            self.x0u = self.episode.x0u
            self.dxu = self.episode.dxu * self.Ksamp

        def _read_data_type_info(self, file_handle):
            self.Ktype = fread(file_handle, 'B')

        def get_data(self, keep_in_mem=False):
            # Check whether data was already read and stored
            if self.data is not None:
                return self.data

            # If not, read the data from the file
            file_handle = open(self.episode.elphy_file.file_name, 'rb')
            data = self._read_data(file_handle)
            if keep_in_mem:
                self.data = data
            return data

        def _read_data(self, file_handle):
            ep = self.episode
            if ep._format_option == 0:
                # Interleaved data: simpler to read the full episode!
                data = ep.get_data()
                # Return the data of this channel
                for i in range(ep.nbchan):
                    if self == ep.channels[i]:
                        return data[i]
            else:
                # Contiguous data: just read the appropriate channel
                offset = ep.data_offset
                offset += sum([ch._data_size for ch in ep.channels[:self.channel_num]])
                file_handle.seek(offset)
                raw = file_handle.read(self._data_size)
                data = np.frombuffer(raw, elphy_types[self.Ktype])
                return data

    class Spikes:
        def __init__(self, ep):
            # Pointer to parent episode
            self.episode = ep

            # No spikes until _read_spikes or _read_waveforms will be called
            self.n_chan = 0
            self.n_spikes_per_chan = []
            self.times = self.units = None
            self.waves = None

        def _read_spikes(self, file_handle, block_size, read_spikes):
            # Header info
            header_size, date = _read_rdata_header(file_handle)
            self.dxu, self.x0u = fread(file_handle, '< d d')
            self.unitX = _read_string(file_handle, max_size=10)
            self.n_chan = fread(file_handle, '< i')
            self.n_spikes_per_chan = fread(file_handle, '< i', self.n_chan)

            # data
            self.spikes_offset = file_handle.tell()
            if read_spikes:
                self.times, self.units = self._read_spikes_data(file_handle)

        def get_data(self, keep_in_mem=False, do_spikes=True, do_waves=True):
            # Check whether data was already read and stored
            if self.times or self.n_chan == 0:
                times, units, waves = self.times, self.units, self.waves
            else:
                # If not, read the data from the file
                ep = self.episode
                file_handle = open(ep.elphy_file.file_name, 'rb')
                times = units = waves = None
                if do_spikes:
                    file_handle.seek(self.spikes_offset)
                    times, units = self._read_spikes_data(file_handle)
                    if keep_in_mem:
                        self.times, self.units = times, units
                if do_waves:
                    file_handle.seek(self.waves_offset)
                    waves = self._read_waves_data(file_handle)
                    if keep_in_mem:
                        self.waves = waves

            if do_spikes and do_waves:
                return times, units, waves
            elif do_spikes:
                return times, units
            elif do_waves:
                return waves
            else:
                return None  # this case should not happen

        def _read_spikes_data(self, file_handle):
            times = [None] * self.n_chan
            spk_units = [None] * self.n_chan
            for i in range(self.n_chan):
                tt = fread(file_handle, '< I', self.n_spikes_per_chan[i])
                times[i] = self.x0u + self.dxu * tt
                spk_units[i] = fread(file_handle, 'B', self.n_spikes_per_chan[i])
            return times, spk_units

        def _read_waves(self, file_handle, block_size, read_spikes):
            # Header info
            self.header_size, self.date = _read_rdata_header(file_handle)
            self.dxu, self.x0u = fread(file_handle, '< d d')  # will overwrite
            # values read by _read_spikes, we should check that they are the
            # same...
            self.unitX = _read_string(file_handle, max_size=10)
            self.dyu, self.y0u = fread(file_handle, '< d d')
            self.unitY = _read_string(file_handle, max_size=10)
            self.dxSource = fread(file_handle, '< d')
            self.unitSource = _read_string(file_handle, max_size=10)
            y_type = fread(file_handle, 'B')
            if y_type != 2:
                raise Exception('Waveform data: only integers are handled')
            self.n_samp, self.n_pretrig = fread(file_handle, '< i i')
            self.n_chan = fread(file_handle, '< i')
            self.n_spikes_per_chan = fread(file_handle, '< i', self.n_chan)

            # data
            self.waves_offset = file_handle.tell()
            if read_spikes:
                self.waves = self._read_waves_data(file_handle)

        def _read_waves_data(self, file_handle):
            wave_size = 24 + 2 * self.n_samp  # 24 corresponds to some additional info (time, etc.) which we won't keep
            waves = [None] * self.n_chan
            for i in range(self.n_chan):
                datai = np.zeros((self.n_spikes_per_chan[i], self.n_samp),
                                 dtype='h')
                for j in range(self.n_spikes_per_chan[i]):
                    file_handle.read(24)  # skip waveform header info
                    datai[j] = fread(file_handle, '< h', self.n_samp)
                datai = self.y0u + self.dyu * datai
                waves[i] = datai
            return waves

    class Memo:
        def __init__(self, f, block_end):
            while f.tell() < block_end:
                ID, SZ, pos1 = _sub_block_ID(f)
                if ID == 'IDENT1':
                    self.name = fread(f, 'str', SZ)
                elif ID == 'ST':
                    self.memo = fread(f, 'str', SZ)
                    self.memo = self.memo.split('\r\n')[:-1]  # list of lines
                f.seek(pos1)

    class Vector:
        def __init__(self, f, block_end):
            while f.tell() < block_end:
                ID, SZ, pos1 = _sub_block_ID(f)
                if ID == 'IDENT1':
                    self.name = fread(f, 'str', SZ)
                elif ID == 'OBJINF':
                    self.tpNum = fread(f, 'b')
                    self.imin = fread(f, 'i')
                    self.imax = fread(f, 'i')
                    self.jmin = fread(f, 'i')  # not used for Tvector
                    self.jmax = fread(f, 'i')  # not used for Tvector
                    self.x0u = fread(f, 'd')
                    self.dxu = fread(f, 'd')
                    self.y0u = fread(f, 'd')
                    self.dyu = fread(f, 'd')
                    self.data = None  # will be read later, by _read_data method
                f.seek(pos1)

        def _read_data(self, f, block_end):
            f.seek(1,
                   1)  # we must skip one byte, i just wrote to Gerard to ask why!!
            self.data = fread(f, elphy_types[self.tpNum],
                              self.imax - self.imin + 1)

    class DBrecord:
        def __init__(self, f, block_end):
            while f.tell() < block_end:
                ID, SZ, pos1 = _sub_block_ID(f)
                if ID == 'IDENT1':
                    self.name = fread(f, 'str', SZ)
                elif ID == 'ST':
                    names = fread(f, 'str', SZ)
                    names = names.split("\r\n")[:-1]
                    npar = len(names)
                elif ID == 'BUF':
                    values = []
                    for i in np.arange(npar):
                        typee = fread(f, 'b')

                        if typee == 1:  # gvBoolean (8 bits, only the first used)
                            values.append(fread(f, '?'))
                        elif typee == 2:  # gvInteger (64 bits)
                            values.append(fread(f, 'q'))
                        elif typee == 3:  # gvFloat (i.e. 'extended' = 10 bytes = 80 bits)
                            values.append(_read_extended(f))
                        elif typee == 4:  # gvString
                            st = _read_string(f, size_bin_type='I')
                            st.replace("\\\\", "\\")
                            values.append(st)
                        # elif typee == 5:  # gvDateTime
                        #     pass
                        # elif typee == 6:  # gvObject
                        #     pass
                        elif typee == 7:  # gvComplex (2*80 bits)
                            val = [0, 0]
                            for j in range(2):
                                x = fread(f, 'Q') * 2 ** -63
                                s_e = fread(f, 'h')
                                s = np.sign(s_e)
                                e = abs(s_e) - (2 ** 14 - 1)
                                val[j] = np.sign(s_e) * x * 2 ** e
                            values.append(val[0] + 1j * val[1])
                        elif typee == 8:  # gvDouble (64 bits)
                            values.append(fread(f, 'd'))
                        elif typee == 9:  # gvDComplex (2*64 bits)
                            values.append(fread(f, 'd') + 1j * fread(f, 'd'))
                        else:
                            print("Unknown data type flag in DBrecord: " + str(
                                typee) + ", cannot proceed reading the record.")
                            values = values + ['cannot be read'] * (npar - i)
                            break
                f.seek(pos1)
            f.seek(block_end)
            self.dict = {names[i]: values[i] for i in np.arange(npar)}


def _read_behavior_parameters(xpar, lines):
    param, is_unique = {}, {}
    for line in lines:
        items = line.split(";")
        name = items[0]
        is_unique[name] = (len(items) < 3) or len(items[2]) == 0
        if is_unique[name]:
            if len(items) < 2:
                values = ''
            else:
                values = items[
                         1:2]  # take only the first value, but remain a list for now
        else:
            values = items[1:]
        for j, val in enumerate(values):
            if val.lower() == 'false':
                values[j] = False
            elif val.lower() == 'true':
                values[j] = True
            else:
                try:
                    values[j] = float(val)
                except:
                    pass
        param[name] = values
    names = param.keys()
    values = param.values()
    is_unique = is_unique.values()
    xpar['fix'] = {name: value[0] for name, value, unique in
                   zip(names, values, is_unique) if unique}
    xpar['table'] = {name: np.array(value) for name, value, unique in
                     zip(names, values, is_unique) if not unique}
    return None


def read_behavior(fname):
    ef = ElphyFile(fname, read_data=True)

    nt, = ef.episodes[0].channels[0].data.shape
    rec = np.zeros((ef.n_episodes, nt))
    dates = [None] * ef.n_episodes
    ep_info = [{}] * ef.n_episodes
    vectors, xpar = {}, {}
    menu_par = None

    # loop over episodes
    for i in range(-1, ef.n_episodes):
        if range == -1:
            objects = ef.initial_objects
        else:
            ep = ef.episodes[i]
            rec[i] = ep.channels[0].data
            dates[i] = ep.date
            objects = ep.objects

        # loop over objects belonging to this episode
        for obj in objects:
            if isinstance(obj, ElphyFile.DBrecord):
                if i == -1:
                    print("Ignoring DBrecord before first episode", obj.dict)
                elif obj.name == 'PG0.MPAR2':
                    # Parameters used to be saved in a DBrecord, now they
                    # are saved in Memo table
                    if menu_par is None:
                        menu_par = obj.dict
                    else:
                        print("Problem ! Several set of parameters in file")
                elif obj.name == 'PG0.EPREPORT':
                    if ep_info[i] == {}:
                        ep_info[i] = obj.dict
                    else:
                        print('Several info DBrecord in episode', i, '!')
                        print('-> Keeping:', ep_info[i])
                        print('-> Ignoring:', obj.dict)
                else:
                    raise Exception(
                        'Unexpected DBrecord with name ' + obj.name)
            elif isinstance(obj, ElphyFile.Vector):
                vectors[obj.name[4:]] = obj.data
            elif isinstance(obj, ElphyFile.Memo):
                if obj.name == 'PG0.PPAR2':
                    _read_behavior_parameters(xpar, obj.memo)
                elif obj.name == 'PG0.IMAGELIST':
                    xpar['imagelist'] = obj.memo
                elif obj.name == 'PG0.SOUNDLIST':
                    xpar['soundlist'] = obj.memo
                else:
                    raise Exception('Unexpected memo with name ' + obj.name)
            else:
                print('Object of type ' + str(type(obj)) + ' not handled by function read_beavior')

    # Merge dates and ep_info into vectors
    for key in ep_info[0].keys():
        vectors[key] = np.array([x[key] for x in ep_info])
    vectors['dates'] = dates

    # Merge menu_par into xpar
    if menu_par:
        if ('fix' in xpar) and xpar['fix']:
            raise Exception(
                'It is unexpected that both menu_par and xpar are defined')
        xpar['fix'] = menu_par

    return rec, vectors, xpar


def elphy_to_klusta(elphy_name, klusta_name=None, ep_nums='all', chan_nums=range(64), ep_len_file=False, junction_len=None):
    """
    Elphy format to klusta-compatible binary format conversion

    Parameters:
    - elphy_name : name of Elphy file to be loaded, or ElphyFile object
    - klusta_name : name of klusta-compatible binary file to generate (if none : defined from elphy_name)
    - ep_nums : list of episode numbers (0 = first) to save to binary file (if 'all': all episodes)
    - chan_nums : list of channels numbers (0 = first) to save to binary file (if 'all': all channels) /!\ all saved channels shall have the same length in a given episode

    Generates 3 files :
    - A klusta-compatible binary file containing the required episodes and channels from the provided Elphy file
    - A text file (xxx_eplengths.txt) with each episode's number of samples (if ep_len_file is True)
    - A text file (xxx_epjunctions.txt) with positions of episode junctions (with a margin of junction_len samples, in junction_len is not None)
    """

    # Load Elphy file if needed
    if isinstance(elphy_name, ElphyFile):
        ef = elphy_name
        elphy_name = ef.file_name
    else:
        ef = ElphyFile(elphy_name, read_data=False)

    if klusta_name == None:
        klusta_name = elphy_name.split('.')[0] + '_binary.dat'

    if ep_nums == 'all':
        ep_nums = range(ef.n_episodes)
    if chan_nums == 'all':
        chan_nums = range(len(ef.episodes[0].channels))

    with open(klusta_name, 'wb') as f:  # Initialize binary file        
        # Do elphy -> klusta conversion for each episode
        for iep in ep_nums:
            ch_lengths=np.array([ef.episodes[iep].channels[ichan].nsamp for ichan in chan_nums])
            assert np.all(ch_lengths[chan_nums] == ch_lengths[chan_nums[0]]) # Check that all required channels have the same length in a given episode
            print('Converting episode ' + str(iep))
            ep_vecs = ef.episodes[iep].get_data()  # Load episode data
            ep_vecs = np.array([ep_vecs[i_elec] for i_elec in chan_nums]).T.flatten()  # Select required channels
            f.write(ep_vecs.astype('<h').tobytes())  # Write data to binary file

    ep_lengths=np.array([ef.episodes[iep].channels[chan_nums[0]].nsamp for iep in ep_nums])
    if ep_len_file:            
        # Write episode lengths to text file        
        with open(klusta_name.split('.')[0] + '_eplengths.txt','w') as f:
            f.write('\n'.join([str(l) for l in ep_lengths]))  
    if junction_len is not None:
        # Write episode junctions to text file
        junctions=np.cumsum(ep_lengths)
        junctions=np.array([np.maximum(junctions[:-1]-junction_len/2,0),np.minimum(junctions[:-1]+junction_len/2,junctions[-1])]).T    
        with open(klusta_name.split('.')[0] + '_epjunctions.txt','w') as f:  # Initialize text file
            f.write('\n'.join([str(start)+' '+str(stop) for start,stop in junctions]))  # Write episode length to text file    # REPRENDRE ICI

            
def get_raw_signal_per_channel_and_episode(ElphyFile, iepisode, ichannel):
    return ElphyFile.episodes[iepisode].channels[ichannel].get_data()*ElphyFile.episodes[iepisode].channels[ichannel].dyu


def get_time_array_per_episode(ElphyFile):
    return np.arange(len(ElphyFile.episodes[0].channels[0].get_data()))*ElphyFile.episodes[0].channels[0].dxu


def realign_episode_data_over_time(ElphyFile, ichannel, subsampling=1):
    """
    subsampling in ms
    """
    isubsmpl = int(subsampling/ElphyFile.episodes[0].channels[0].dxu)
    print(isubsmpl)
    linear_data = np.zeros(0, dtype=np.int16)
    for iep in range(len(ElphyFile.episodes)):
        linear_data = np.concatenate([linear_data, np.array(ElphyFile.episodes[iep].channels[ichannel].get_data(), dtype=np.int16)[::isubsmpl]])
        
    return linear_data

if __name__ == "__main__":

    if len(sys.argv)<2:
        print('need to give a datafile as an argument, e.g.: "python elphy_reader.py /media/yzerlaut/Elements/4319_CXRIGHT/4319_CXRIGHT_NBR1.DAT" ')
    else:
        filename = sys.argv[-1]
        ef = ElphyFile(filename, read_data=False, read_spikes=False)

        new_dt = 10 # ms
        subsmpl_data = realign_episode_data_over_time(ef, 5, subsampling=new_dt)
        new_t = np.arange(len(subsmpl_data))*new_dt*1e-3 # in seconds
        
        # t = get_time_array_per_episode(ef) # in
        import matplotlib.pylab as plt

        plt.plot(new_t, subsmpl_data)
        plt.show()
        
        # for ax, iepisode in zip(AX.flatten(), [0, 4, 6, 34]):
        #     ax.set_title('episode %i' % iepisode)
        #     for ichannel in [3,45, 23, 12]:
        #         ax.plot(t[::100], get_raw_signal_per_channel_and_episode(ef, iepisode, ichannel)[::100])
        #     ax.set_ylabel(ef.episodes[0].channels[0].uY)
        #     ax.set_xlabel(ef.episodes[0].channels[0].uX)
        # plt.show()

        # Episodes = range(10,20)+range(200,240)

        # for iepisode in Episodes:
        #     raw_episode = np.array([
        #         get_raw_signal_per_channel_and_episode(ef, iepisode, ichannel) for ichannel in range(67)])

