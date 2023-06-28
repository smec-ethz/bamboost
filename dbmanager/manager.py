from ctypes import ArgumentError
from mpi4py import MPI
import os
import shutil
import uuid
import h5py
import pandas as pd

from .simulation import Simulation, SimulationReader, SimulationWriter
from .common.file_handler import open_h5file


class Manager:
    """Data Manager.

    Args:
        path (str): path to the directory of the database. If doesn't exist,
            a new database will be created.
        comm: MPI communicator
    """

    def __init__(self, path: str, comm: MPI.Comm = MPI.COMM_WORLD):
        self.path = path
        self.comm = comm
        os.makedirs(self.path, exist_ok=True)

    def __getitem__(self, row: int):
        """Returns the simulation in the specified row of the dataframe.

        Args:
            row (int): return the simulation with index specified by row
        Returns:
            sim (SimulationReader)
        """
        return self.sim(self.df.loc[row, 'id'])

    def __repr__(self):
        display(self.df)
        return f'Database {self.path}'

    @property
    def df(self) -> pd.DataFrame:
        """Return pandas dataframe of the database."""
        all_uids = self._get_uids()
        data = list()

        for uid in all_uids:
            with open_h5file(os.path.join(os.path.join(self.path, uid),
                                        f'{uid}.h5'), 'r') as f:
                tmp_dict = dict()
                if 'parameters' in f.keys():
                    tmp_dict.update(f['parameters'].attrs)
                if 'additionals' in f.keys():
                    tmp_dict.update({'additionals': f['additionals'].attrs})
                tmp_dict.update(f.attrs)
                data.append(tmp_dict)

        df = pd.DataFrame.from_records(data)
        if df.empty:
            return df

        # Sort dataframe columns
        df = df[['id', 'notes', *df.columns.difference(['id', 'notes'])]]
        return df

    def sim(self, arg, sort: str = None, reverse: bool = False) -> Simulation:
        """Get an existing simulation with uid.

        Args:
            arg (str): unique identifier
            arg (condition): pandas selection (return tuple if more than one match)
            sort (str): parameter to sort the returned list with
        Returns:
            sim (SimulationReader) or list of sims (list)
        """
        if isinstance(arg, str):
            existing_sim = SimulationReader(arg, self.path, self.comm)
            return existing_sim

        id_list = self.df[arg]['id'].values
        existing_sims = [SimulationReader(uid, self.path, self.comm) for uid in id_list]
        if sort is None:
            return existing_sims

        sorted_sims = sorted(existing_sims, key=lambda s: s.parameters[sort], reverse=reverse)
        return sorted_sims
            
    def sims(self, sort: str = None, reverse: bool = False) -> list:
        """Get all simulations in a list."""
        id_list = self.df['id'].values
        existing_sims = [SimulationReader(uid, self.path, self.comm) for uid in id_list]

        if sort is None:
            return existing_sims
        else:
            return sorted(existing_sims, key=lambda s: s.parameters[sort], reverse=reverse)

    def create_simulation(self, uid: str = None, parameters: dict = None,
                          skip_duplicate_check: bool = False) -> SimulationWriter:
        """Get a writer object for a new simulation. This is written for paralell
        as it is likely that this may be used in an executable creating multiple runs
        for a parametric space, which may be run in paralell.

        Args:
            uid (str): The name/uid for the simulation. If not specified, a random id
                will be assigned.
            parameters (dict): Parameter dictionary. If provided, the parameters will be 
                checked against the existing sims for duplication. Otherwise, they may be 
                specified later with `add_parameters`.
            skip_duplicate_check (bool): if True, the duplicate check is skipped.
        Returns:
            sim (SimulationWriter)
        """
        if parameters and not skip_duplicate_check:
            go_on, _uid = self._check_duplicate(parameters)
            if not go_on:
                print('Aborting by user desire...')
                return None
            if _uid:
                uid = _uid

        if self.comm.rank==0:
            if not uid:
                uid = uuid.uuid4().hex[:8]  # Assign random unique identifier

        uid = self.comm.bcast(uid, root=0)
        new_sim = SimulationWriter(uid, self.path, self.comm)            
        new_sim.create()
        if parameters:
            new_sim.add_parameters(parameters)
        return new_sim

    def remove(self, uid: str) -> None:
        """CAUTION, DELETING DATA. Remove the data of a simulation.

        Args:
            uid (str): id
        """
        shutil.rmtree(os.path.join(self.path, uid))

    def _get_uids(self) -> list:
        """Get all simulation names in the database."""
        return [dir for dir in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, dir))]

    def _check_duplicate(self, parameters) -> tuple:
        """Checking whether the parameters dictionary exists already.
        May need to be improved...

        Args:
            parameters (dict): parameter dictionary to check for
        Returns:
            Tuple(Bool, uid) wheter to continue and with what uid.
        """
        duplicates = list()

        for uid in self._get_uids():
            with open_h5file(os.path.join(os.path.join(self.path, uid),
                                        f'{uid}.h5'), 'r') as f:
                if 'parameters' not in f.keys():
                    continue

                tmp_dict = dict()
                tmp_dict.update(f['parameters'].attrs)

                if tmp_dict==parameters:
                    duplicates.append((uid, 'equal'))
                    continue

                shared_keys = (tmp_dict.keys() & parameters.keys())
                if ({key: tmp_dict[key] for key in shared_keys}
                    =={key: parameters[key] for key in shared_keys}):
                    duplicates.append((uid, 'shared_equal'))
                    continue

        if not duplicates:
            return True, None

        # What should be done?
        print(f"The parameter space may already exist. Here are the duplicates:", flush=True)
        display(self.df[self.df['id'].isin([i[0] for i in duplicates])])
        
        prompt = input("Create with altered uid (`c`), Create with new id (`n`), Abort (`a`)")
        if prompt=='a':
            return False, None
        if prompt=='n':
            return True, None
        if prompt=='c':
            return True, self._generate_subuid(duplicates[0][0].split('.')[0])

        raise ArgumentError('Answer not valid! Aborting')

    def _generate_subuid(self, uid_base: str) -> str:
        """Return a new sub uid for the base uid.
        Following the following format: `base_uid.1`

        Args:
            uid_base (str): base uid for which to find the next subid.
        Returns:
            New uid string
        """
        uid_list = [uid for uid in self._get_uids() if uid.startswith(uid_base)]
        subiterator = max([int(id.split('.')[1]) for id in uid_list if len(id.split('.'))>1] + [0])
        return f'{uid_base}.{subiterator+1}'


        

