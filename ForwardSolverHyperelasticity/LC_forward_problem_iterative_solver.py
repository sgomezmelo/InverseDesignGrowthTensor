from fenics import *
import dolfin
import numpy as np
from ufl_legacy import nabla_div, VectorElement, FiniteElement, MixedElement, split, replace, cos, sin
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
IterativeNewtonSolver(TargetState,TargetState,J,dE,SolutionSpace,1)
print('Solved target problem')
u_target, bla, blah = split(TargetState)
#F_target = nabla_grad(u_target)+Identity(d)

uShow, mm, pp = TargetState.split()
with XDMFFile(dolfin.MPI.comm_world, save_dir+mesh_name+"/targetDeformation.xdmf") as xdmf_outfile:
    xdmf_outfile.write(uShow)



