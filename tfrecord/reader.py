"""Reader utils."""

import functools
import io
import os
import struct
import typing

import numpy as np

from tfrecord import example_pb2
from tfrecord import iterator_utils

# NOTE(ycho): gcsfs DOES NOT have to be from `torch_xla`.
# Consider removing this awkward dependency...
from torch_xla.utils import gcsfs


def tfrecord_iterator(data_path: str,
                      index_path: typing.Optional[str] = None,
                      shard: typing.Optional[typing.Tuple[int, int]] = None,
                      gcs:bool = False
                      ) -> typing.Iterable[memoryview]:
    """Create an iterator over the tfrecord dataset.

    Since the tfrecords file stores each example as bytes, we can
    define an iterator over `datum_bytes_view`, which is a memoryview
    object referencing the bytes.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str, optional, default=None
        Index file path. Can be set to None if no file is available.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    datum_bytes_view: memoryview
        Object referencing the specified `datum_bytes` contained in the
        file (for a single record).
    """

    # NOTE(ycho): Bookkeeping for whether we should close the file after we're done.
    should_close = True
    if isinstance(data_path, io.IOBase):
        # NOTE(ycho): Deal with supplied file-like object arg.
        file = data_path
        should_close = False
    else:
        if gcs:
            # NOTE(ycho): internally, this function just downloads the file and actually
            # just returns BytesIO instead. But then again, we'd like to access the
            # whole file anyway.
            file = gcsfs.open(data_path, "rb")
        else:
            # load file locally.
            file = io.open(data_path, "rb")

    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)

    def read_records(start_offset=None, end_offset=None):
        nonlocal length_bytes, crc_bytes, datum_bytes

        # If unspecified, query end_offset for `BytesIO` objects.
        if (end_offset is None) and isinstance(file, io.BytesIO):
            pos = file.tell()
            file.seek(0, io.SEEK_END)
            end_offset = file.tell()
            file.seek(pos, io.SEEK_SET)

        if start_offset is not None:
            file.seek(start_offset)

        # Fallback method in case `BytesIO` based one failed.
        if end_offset is None:
            if gcs:
                # Fallback method, requires another request over the network.
                # In general, this part should never run.
                end_offset = gcsfs.stat(data_path).size
            else:
                # Fallback method for regular files.
                end_offset = os.path.getsize(data_path)

        while file.tell() < end_offset:
            if file.readinto(length_bytes) != 8:
                raise RuntimeError("Failed to read the record size.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the start token.")
            length, = struct.unpack("<Q", length_bytes)
            if length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:length]
            if file.readinto(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")
            yield datum_bytes_view

    if index_path is None:
        yield from read_records()
    else:
        index = np.loadtxt(index_path, dtype=np.int64)[:, 0]
        if shard is None:
            offset = np.random.choice(index)
            yield from read_records(offset)
            yield from read_records(0, offset)
        else:
            num_records = len(index)
            shard_idx, shard_count = shard
            start_index = (num_records * shard_idx) // shard_count
            end_index = (num_records * (shard_idx + 1)) // shard_count
            start_byte = index[start_index]
            end_byte = index[end_index] if end_index < num_records else None
            yield from read_records(start_byte, end_byte)

    if should_close:
        file.close()


def process_feature(feature: example_pb2.Feature,
                    typename: str,
                    typename_mapping: dict,
                    key: str):
    # NOTE: We assume that each key in the example has only one field
    # (either "bytes_list", "float_list", or "int64_list")!
    field = feature.ListFields()[0]
    inferred_typename, value = field[0].name, field[1].value

    if typename is not None:
        tf_typename = typename_mapping[typename]
        if tf_typename != inferred_typename:
            reversed_mapping = {v: k for k, v in typename_mapping.items()}
            raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                        f"(should be '{reversed_mapping[inferred_typename]}').")

    if inferred_typename == "bytes_list":
        value = np.frombuffer(value[0], dtype=np.uint8)
    elif inferred_typename == "float_list":
        value = np.array(value, dtype=np.float32)
    elif inferred_typename == "int64_list":
        value = np.array(value, dtype=np.int32)
    return value


def extract_feature_dict(features, description, typename_mapping):
    if isinstance(features, example_pb2.FeatureLists):
        features = features.feature_list

        def get_value(typename, typename_mapping, key):
            feature = features[key].feature
            fn = functools.partial(process_feature, typename=typename,
                                   typename_mapping=typename_mapping, key=key)
            return np.array(list(map(fn, feature)))
    elif isinstance(features, example_pb2.Features):
        features = features.feature

        def get_value(typename, typename_mapping, key):
            return process_feature(features[key], typename,
                                   typename_mapping, key)
    else:
        raise TypeError(f"Incompatible type: features should be either of type "
                        f"example_pb2.Features or example_pb2.FeatureLists and "
                        f"not {type(features)}")

    all_keys = list(features.keys())

    if description is None:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, (list, tuple)):
        description = dict.fromkeys(description, None)

    processed_features = {}
    for key, typename in description.items():
        if key not in all_keys:
            raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")

        processed_features[key] = get_value(typename, typename_mapping, key)

    return processed_features


def example_loader(data_path: str,
                   index_path: typing.Union[str, None],
                   description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                   shard: typing.Optional[typing.Tuple[int, int]] = None,
                   gcs:bool = False
                   ) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str or None
        Index file path. Can be set to None if no file is available.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    features: dict of {str, np.ndarray}
        Decoded bytes of the features into its respective data type (for
        an individual record).
    """

    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    record_iterator = tfrecord_iterator(data_path, index_path, shard, gcs)

    for record in record_iterator:
        example = example_pb2.Example()
        example.ParseFromString(record)

        yield extract_feature_dict(example.features, description, typename_mapping)


def sequence_loader(data_path: str,
                    index_path: typing.Union[str, None],
                    context_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                    features_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                    shard: typing.Optional[typing.Tuple[int, int]] = None,
                    gcs:bool = False
                    ) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    record_iterator = tfrecord_iterator(data_path, index_path, shard, gcs)

    for record in record_iterator:
        example = example_pb2.SequenceExample()
        example.ParseFromString(record)

        context = extract_feature_dict(example.context, context_description, typename_mapping)
        features = extract_feature_dict(example.feature_lists, features_description, typename_mapping)

        yield context, features


def tfrecord_loader(data_path: str,
                    index_path: typing.Union[str, None],
                    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                    shard: typing.Optional[typing.Tuple[int, int]] = None,
                    gcs:bool = False
                    ) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str or None
        Index file path. Can be set to None if no file is available.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    features: dict of {str, np.ndarray}
        Decoded bytes of the features into its respective data type (for
        an individual record).
    """
    return example_loader(data_path, index_path, description, shard, gcs)


def multi_tfrecord_loader(data_pattern: str,
                          index_pattern: typing.Union[str, None],
                          splits: typing.Dict[str, float],
                          description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                          ) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    """Create an iterator by reading and merging multiple tfrecord datasets.

    NOTE: Sharding is currently unavailable for the multi tfrecord loader.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str or None
        Input index path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    Returns:
    --------
    it: iterator
        A repeating iterator that generates batches of data.
    """
    loaders = [functools.partial(tfrecord_loader, data_path=data_pattern.format(split),
                                 index_path=index_pattern.format(split) \
                                     if index_pattern is not None else None,
                                 description=description)
               for split in splits.keys()]
    return iterator_utils.sample_iterators(loaders, list(splits.values()))
