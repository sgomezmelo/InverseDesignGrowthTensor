from fenics import *
import dolfin
import numpy as np
from ufl import nabla_div, VectorElement, FiniteElement, MixedElement, split, replace, cos, sin
import math 
import meshio
import sys
import os
from funcsTensorCalc import *
import matplotlib.pyplot as plt

#set_log_active(False)

#Create result folder and read undeformed geometry 
save_dir = "./Results/"
geom_folder = "./Geometries/"
#geom_subfolder = input("Type name of .msh subfolder path: ")
#mesh_name = input("Type name of .msh file: ")
geom_subfolder = 'Disk2D'
mesh_name = 'annulus_2D_2D'
#g_name = input("Choose director field name: ")
os.system("mkdir "+save_dir+mesh_name)

mesh = dolfin.cpp.mesh.Mesh()
mvc_subdomain = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim())
mvc_boundaries = dolfin.MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)

with XDMFFile(MPI.comm_world, geom_folder+geom_subfolder+'/'+mesh_name+".xdmf") as xdmf_infile:
    xdmf_infile.read(mesh)
    xdmf_infile.read(mvc_subdomain, "")

domains = dolfin.cpp.mesh.MeshFunctionSizet(mesh, mvc_subdomain)
dx = Measure('dx', domain=mesh, subdomain_data=domains)

q_degree = 5
dx = dx(metadata={'quadrature_degree': q_degree}) 

parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
 
d = 2
n_rigid = int(d + d*(d-1)/2)
n_sym = int(d*(d+1)/2)
#n_sym = 3

#Define function space as Vector space (displacement) + scalar function (incompressibility) + 6 constants to eliminate rigid motions
DisplacementElement = VectorElement("CG", mesh.ufl_cell(), 2) 
LagrangeMultiplierCte = FiniteElement("Real", mesh.ufl_cell(), 0)
PressureElement = FiniteElement("CG", mesh.ufl_cell(), 1) 
RigidMotions = MixedElement([LagrangeMultiplierCte for i in range(n_rigid)]) 
mixed_element = MixedElement([DisplacementElement,RigidMotions,PressureElement])
#mixed_element = MixedElement([DisplacementElement,PressureElement])


SolutionSpace = FunctionSpace(mesh,mixed_element) 
#AngleSpace = FunctionSpace(mesh,AngleElement) # iteration = 0

#TensorSpace = FunctionSpace(mesh,TensorCompME)
State = Function(SolutionSpace)
#State.vector()[:] = np.random.random(State.vector()[:].shape)*1e-2
TargetState = Function(SolutionSpace)
adjoint_state =  Function(SolutionSpace)
K0 = 0.5
# #nT = Expression(('cos(t)','sin(t)',0.0), t = polarAngle,pi = np.pi, degree = 1)
λT = 0.8
polarAngle = Expression('atan2(x[1],x[0])', degree = 1)
nT = Expression(('cos(t)','sin(t)'), t = polarAngle,pi = np.pi, degree = 1)
#nT = as_vector((1.0,0.0))
#nT = RadialConstantCurvature(K0, λT, 1/λT**(1/(d-1)))
InverseGrowthTensor = (1/λT -λT**(1/(d-1)))*outer(nT,nT)+λT**(1/(d-1))*Identity(d)
dE = derivative(NeoHookeanEnergy(TargetState,InverseGrowthTensor,d = d)*dx,TargetState)
J = derivative(dE,TargetState)
#SolveNonLinearProblem(dE, TargetState, J,[],ffc_options, linear_solver = 'mumps',preconditioner = None,initial_relaxation = 0.5)
#NewtonSolver(TargetState,TargetState,J,dE,SolutionSpace,PC(TrialFunction(SolutionSpace),TestFunction(SolutionSpace),dx),1.0)
#def IterativeNewtonSolver(u0,func,Jacobian,F,SolutionSpace,τ):
dofs_u = SolutionSpace.sub(0).dofmap().dofs()
#ksp = IterativeNewtonSolver(TargetState,TargetState,J,dE,SolutionSpace,1)
#print('Solved target problem')
#u_target, bla, blah = split(TargetState)
##F_target = nabla_grad(u_target)+Identity(d)
#
#uShow, mm, pp = TargetState.split()
#with XDMFFile(dolfin.MPI.comm_world, save_dir+mesh_name+"/targetDeformation.xdmf") as xdmf_outfile:
#    xdmf_outfile.write(uShow)
#
#
#

###

#def IterativeNewtonSolver(u0,func,Jacobian,F,SolutionSpace,τ):
#ksp = IterativeNewtonSolver(TargetState,TargetState,J,dE,SolutionSpace,1)
error = 1
tau = 1
tol = 1e-5
i = 0
max_iter = 100
u_i = Function(SolutionSpace)
u_i.vector()[:] = TargetState.vector()[:]
J_i = J # replace(J,{func:u_i})
F_i = dE #replace(dE,{func:u_i})

# assemble system
A_fenics, b_fenics = assemble_system(J_i, F_i)
A = as_backend_type(A_fenics).mat()
b = as_backend_type(b_fenics).vec()

dofs_u = SolutionSpace.sub(0).dofmap().dofs()
dofs_c = SolutionSpace.sub(1).dofmap().dofs()
dofs_p = SolutionSpace.sub(2).dofmap().dofs()

dofs_lambda = list(dofs_c) + list(dofs_p)
#dofs_lambda = list(dofs_p)

rstart, rend = A.getOwnershipRange()

local_u = [i for i in dofs_u if rstart <= i < rend]
local_lambda = [i for i in dofs_lambda if rstart <= i < rend]

is_u = PETSc.IS().createGeneral(local_u, comm=PETSc.COMM_WORLD)
is_lambda = PETSc.IS().createGeneral(local_lambda, comm=PETSc.COMM_WORLD)

update = Function(SolutionSpace)




#print("Assembled A and c")
###
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.setTolerances(rtol=1e-8)

pc = ksp.getPC()
pc.setType(PETSc.PC.Type.FIELDSPLIT)

pc.setFieldSplitIS(("u", is_u), ("p", is_lambda))
pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

# Force PETSc to build sub-KSPs
ksp.setUp()
print("done!")

###


ksp = PETSc.KSP().create()
ksp.setOperators(A)
#ksp.setType("gmres")
ksp.setType(PETSc.KSP.Type.GMRES)
ksp.setTolerances(rtol=1e-8)

pc = ksp.getPC()
pc.setType(PETSc.PC.Type.FIELDSPLIT)

pc.setFieldSplitIS(
    ("u", is_u),
    ("lamb", is_lambda)
)

pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)

pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

# pc.setFromOptions()
# pc.setUp()

ksp.setUp()



#Nullspace on Multiplier block:
# null_vec = A.createVecRight()
# null_vec.set(0.0)

# arr = null_vec.getArray()
# arr[dofs_lambda] = 1.0
# null_vec.resetArray()

# null_vec.normalize()

# ns = PETSc.NullSpace().create(vectors=[null_vec])
# A.setNullSpace(ns)
ksp.solve(b,update.vector().vec())

#Update
u_i.vector()[:] = u_i.vector()[:] - tau*update.vector()[:] #Update Solution
#u_i.assign(u_i - τ*update)
#F_i = replace(F,{func:u_i})
error = sum(assemble(F_i) * assemble(F_i))
print(error)
#
#print(sum(update[:] * update[:]))


i += 1

##
