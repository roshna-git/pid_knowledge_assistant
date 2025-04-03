Starting pipeline optimization...
urllib3\connectionpool.py:1099: InsecureRequestWarning: Unverified HTTPS request is being made to host 'sv03919.res1.rlaone.net'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
Working directory set to: C:\Users\IDM286398\Downloads\PathFinder_Dir (1)\PathFinder_Dir\_internal
Found config.json in bundled directory
Keeping user's output directory: C:/Users/IDM286398/Desktop/output
Updated config.json with correct paths
Validating license with server: https://sv03919.res1.rlaone.net/license
License validation result: True
All elevations match within 0.1 m.
Generating interpolated terrain...
Smoothing terrain...
Generating initial burial profile...
Generating initial cross profile...
Optimizing pipeline profile...
WARNING: Failed to create solver with name 'ipopt': Failed to set executable
for solver ipopt. File with name=C:\bilfinger\pathfinder\ipopt\bin\ipopt.exe
either does not exist or it is not executable. To skip this validation, call
set_executable with validate=False.
Traceback (most recent call last):
  File "pyomo\opt\base\solvers.py", line 148, in __call__
    opt = self._cls[_name](**kwds)
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "pyomo\solvers\plugins\solvers\IPOPT.py", line 44, in __init__
    super(IPOPT, self).__init__(**kwds)
  File "pyomo\opt\solver\shellcmd.py", line 66, in __init__
    self.set_executable(name=executable, validate=validate)
  File "pyomo\opt\solver\shellcmd.py", line 115, in set_executable
    raise ValueError(
ValueError: Failed to set executable for solver ipopt. File with name=C:\bilfinger\pathfinder\ipopt\bin\ipopt.exe either does not exist or it is not executable. To skip this validation, call set_executable with validate=False.
Traceback (most recent call last):
  File "main.py", line 208, in <module>
  File "main.py", line 82, in run_pipeline_optimization
  File "pyomo_router.py", line 224, in generate_optimal_profile
    if not solver.available():
           ^^^^^^^^^^^^^^^^^^
  File "pyomo\opt\base\solvers.py", line 86, in available
    raise ApplicationError("Solver (%s) not available" % str(self.name))
pyomo.common.errors.ApplicationError: Solver (ipopt) not available
