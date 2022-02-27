__all__ = [
    'ZipFolderDatasetCreator',
    'MultipleSamples',
]

from zutils.indexed_record import dumps_single_object, loads_single_object
from zutils.zipfile_storage import ZipFileStorage
from z_python_utils.classes import global_registry
from z_python_utils.io import mkdir_p, robust_remove
from typing import Sized
from .async_executors import MCPThreadPoolExecutor, ProcessPoolExecutorWithProgressBar
from uuid import uuid4
import os
import json
import datetime


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
            epoch_uuid = datetime.datetime.now().isoformat().replace(":", "") + "-" + str(uuid4())
            epoch_prefix = os.path.join(self.folder_path, epoch_uuid)
            epoch_zipfile = ZipFileStorage(
                epoch_prefix + ".zip", 'w',
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
            num_micro_samples = len(epoch_zipfile)
            with open(epoch_prefix + ".meta.json", 'w') as f:
                json.dump(dict(
                    num_micro_samples=num_micro_samples
                ), f)
            with open(epoch_prefix + ".complete", "w") as f:
                print(datetime.datetime.now().isoformat(), file=f)
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
        self.folder_path = folder_path
        with open(os.path.join(self.folder_path, "meta.json"), "r") as f:
            m = json.load(f)
        # macro meta
        self.num_macro_samples = m['num_macro_samples']
        self.epoch_filenames = [
            x[:-len('.complete')] for x in filter(lambda x: x.endswith('.complete'), os.listdir(self.folder_path))
        ]
        # micro meta
        self.num_micro_samples = []
        for epoch_fn in self.epoch_filenames:
            with open(os.path.join(self.folder_path, epoch_fn + ".zip"), "r") as f:
                sm = json.load(f)
            self.num_micro_samples.append(sm['num_micro_samples'])
        if not self.num_micro_samples:
            self.num_samples = self.num_macro_samples
        elif len(set(self.num_micro_samples)) == 1:
            self.num_samples = self.num_micro_samples[0]
        else:
            self.num_samples = None
        self.total_samples = sum(self.num_micro_samples)
        self._current_epoch = 0
        self._currently_visited_sample_ids = set()

    @property
    def total_epochs(self) -> int:
        return len(self.epoch_filenames)

    @property
    def current_epoch(self) -> int:
        return self._current_epoch % self.total_epochs

    def set_epoch(self, epoch_id: int):
        self._current_epoch = epoch_id % self.total_epochs
        self._currently_visited_sample_ids = set()

    def __len__(self) -> int:
        if self.num_samples is not None:
            return self.num_samples
        else:
            return self.num_micro_samples[self.current_epoch]

    def __iter__(self):
        current_epoch = self.current_epoch
        for i in range(len(self)):
            yield self.get_item(current_epoch, i)
        self.set_epoch(self.current_epoch + 1)

    def __getitem__(self, i):
        if i in self._currently_visited_sample_ids:
            self.set_epoch(self.current_epoch + 1)
        return self.get_item(self.current_epoch, sample_id=0)

    def get_item(self, epoch_id, sample_id):
        pass

