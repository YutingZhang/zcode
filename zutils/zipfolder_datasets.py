__all__ = [
    'ZipFolderDatasetCreator',
    'MultipleSamples',
]

from zutils.indexed_record import dumps_single_object, loads_single_object
from zutils.zipfile_storage import ZipFileStorage
from z_python_utils.classes import global_registry
from z_python_utils import mkdir_p
from typing import Sized
from .async_executors import MCPThreadPoolExecutor, ProcessPoolExecutorWithProgressBar
from uuid import uuid4
import os
import json


class ZipFolderDatasetCreator:
    def __init__(self, folder_path: str, num_macro_samples: int):
        self.folder_path = folder_path
        if os.path.exists(self.meta_file_path):
            with open(self.meta_file_path, 'r') as f:
                m = json.load(f)
            assert m['num_macro_samples'] == num_macro_samples, "mismatched num_macro_samples"
        else:
            with open(self.meta_file_path, 'w') as f:
                json.dump(dict(
                    num_macro_samples=num_macro_samples
                ), f)
        self.num_macro_samples = num_macro_samples
        mkdir_p(self.folder_path)

    @property
    def meta_file_path(self) -> str:
        return os.path.join(self.folder_path, "meta.json")

    def generate(self, orig_dataset, num_epochs: int, num_workers: int):
        n = self.num_macro_samples
        if isinstance(orig_dataset, Sized):
            n = len(orig_dataset)
            assert n == self.num_macro_samples, "raw"
        for epoch in range(num_epochs):
            print("Epoch: %d / %d" % (epoch+1, num_epochs))
            epoch_uuid = str(uuid4())
            epoch_zipfile = ZipFileStorage(
                os.path.join(self.folder_path, epoch_uuid + ".zip"), 'w',
                serialization_func=dumps_single_object, deserialization_func=loads_single_object
            )
            per_sample_executor = ProcessPoolExecutorWithProgressBar(num_workers=num_workers, num_tasks=n)
            with global_registry(orig_dataset) as dataset_registry, global_registry(epoch_zipfile) as zipfile_registry:
                if num_workers > 0:
                    zipfile_executor = MCPThreadPoolExecutor(max_workers=1)
                else:
                    zipfile_executor = None
                for i in range(n):
                    per_sample_executor.submit(
                        _get_and_store_samples, i=i,
                        dataset_registry=dataset_registry, zipfile_registry=zipfile_registry,
                        zipfile_executor=zipfile_executor
                    )
            per_sample_executor.join()
            per_sample_executor.shutdown()
            del per_sample_executor
            if zipfile_executor is not None:
                zipfile_executor.shutdown()
            epoch_zipfile.close()


class MultipleSamples(list):
    pass


def _write_to_zip_file(zipfile_registry, d, i: int):
    epoch_zipfile = global_registry[zipfile_registry]
    if isinstance(d, MultipleSamples):
        for j, r in enumerate(d):
            entry_id_str = "{:d}-{:d}".format(i, j)
            epoch_zipfile[entry_id_str] = r
    else:
        entry_id_str = str(i)
        epoch_zipfile[entry_id_str] = d


def _get_and_store_samples(i, dataset_registry, zipfile_registry, zipfile_executor):
    dataset = global_registry[dataset_registry]
    d = dataset[i]
    if zipfile_executor is None:
        _write_to_zip_file(zipfile_registry, d, i)
    else:
        fr = zipfile_executor.submit(_write_to_zip_file, zipfile_registry, d, i)
        fr.result()


class ZipFolderDataset:
    def __init__(self, folder_path: str):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def num_epochs(self):
        pass

    def get_item(self, epoch_id, sample_id):
        pass

