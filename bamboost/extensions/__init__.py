class ExtensionsLazyLoader:
    # This pattern is used to provide a single point/namespace to access all
    # the extensions without importing them directly.
    # 
    # TODO: this is the best way I could come up with, please suggest a better
    # approach if you have one.

    @property
    def FenicsWriter(self):
        from .fenics import FenicsWriter
        return FenicsWriter

    @property
    def Remote(self):
        from .remote_manager import Remote
        return Remote

    @property
    def RemoteManager(self):
        from .remote_manager import RemoteManager
        return RemoteManager

    @property
    def install_slurm(self):
        from .slurm import install
        return install


extensions = ExtensionsLazyLoader()
