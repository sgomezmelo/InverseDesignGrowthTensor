from fenics import *
import dolfin
import numpy as np
from ufl_legacy import nabla_div, VectorElement, FiniteElement, MixedElement, split, replace
import math 
import meshio
import sys
import os
from dolfin import PETScOptions
from petsc4py import PETSc

parameters["linear_algebra_backend"] = "PETSc"
parameters["reorder_dofs_serial"] = True

def assemble_sym_tensor(TensorComps,d,TensorInSurface=True):
    
    if d == 3: 
      if TensorInSurface:
        T = as_tensor(((TensorComps[0],TensorComps[1],0),
               (TensorComps[1],TensorComps[2],0),
               (0,0,1)))
      
      else:
        T = as_tensor(((TensorComps[0],TensorComps[1],TensorComps[2]),
               (TensorComps[1],TensorComps[3],TensorComps[4]),
               (TensorComps[2],TensorComps[4],TensorComps[5])))
    else:
       T = as_tensor(((TensorComps[0],TensorComps[1]),
               (TensorComps[1],TensorComps[2])))
       
    
    return T

def NeoHookeanEnergy(q,InverseGrowthTensor,d = 3):
    u, c, p = split(q)
    if d == 3:
      e1 = Constant((1.0,0.0,0.0))
      e2 = Constant((0.0,1.0,0.0))
      e3 = Constant((0.0,0.0,1.0))
      e4 = Expression(('-x[1]','x[0]',0.0), degree = 1)
      e5 = Expression(('-x[2]',0.0,'x[0]'), degree = 1)
      e6 = Expression((0.0,'-x[2]','x[1]'), degree = 1)
      e = [e1,e2,e3,e4,e5,e6]
    
    else:
      e1 = Constant((1.0,0.0))
      e2 = Constant((0.0,1.0))
      e4 = Expression(('-x[1]','x[0]'), degree = 1)
      e = [e1,e2,e4]
    
        

    #inv_F_n = inv(Fn) #Inverse growth tensor
    F = nabla_grad(u)+Identity(d) 
    A = InverseGrowthTensor*F 
    gramA = dot(A,A.T) 
    w = tr(gramA)/2.0 + p*(det(F)-1.0)
    n_rigid = int(d + d*(d-1)/2)
    for i in range(n_rigid): 
        w += (c[i]*inner(u,e[i]))
    return w
  
def PenaltyNeoHookeanEnergy(q,InverseGrowthTensor,d = 3,kpen=1e-3):
    u, p = split(q)
    if d == 3:
      #e1 = Constant((1.0,0.0,0.0))
      #e2 = Constant((0.0,1.0,0.0))
      #e3 = Constant((0.0,0.0,1.0))
      e4 = Expression(('-x[1]','x[0]',0.0), degree = 1)
      e5 = Expression(('-x[2]',0.0,'x[0]'), degree = 1)
      e6 = Expression((0.0,'-x[2]','x[1]'), degree = 1)
      #e = [e1,e2,e3,e4,e5,e6]
      e = [e4,e5,e6]
    
    else:
      #e1 = Constant((1.0,0.0))
      #e2 = Constant((0.0,1.0))
      e4 = Expression(('-x[1]','x[0]'), degree = 1)
      #e = [e1,e2,e4]
      e = [e4]
        

    #inv_F_n = inv(Fn) #Inverse growth tensor
    F = nabla_grad(u)+Identity(d) 
    A = InverseGrowthTensor*F 
    gramA = dot(A,A.T) 
    w = tr(gramA)/2.0 + p*(det(F)-1.0)+kpen*(inner(u,u))
    
    #n_rigid = int(d*(d-1)/2)

    # for i in range(n_rigid): 
    #     w += (c[i]*inner(u,e[i]))
        
    return w

def RadialConstantCurvature(K_0, alpha_1, alpha_2):
    c_k = K_0/(alpha_1**(-2.0)-alpha_2**(-2.0)) #Some strange curvature dependent constant that we need
    c2 = 1.0-2.0/(1.0+(alpha_1/alpha_2))
    #c2  = 1.0
    theta = Expression('atan2(x[1],x[0])', degree = 1)
    r2 = Expression('pow(x[1],2)+pow(x[0],2)', degree = 1)
    alpha = Expression('0.5*acos(-c_k*r2/2.0 + c2)', r2 = r2, c_k = c_k, c2 = c2, degree = 1)
    n = Expression((('cos(phi+alpha)','sin(phi+alpha)',0.0)), phi = theta, alpha = alpha, degree = 1)
    return n
   
def SolveNonLinearProblem(Variation, func, Jacobian,bc,ffc_options, linear_solver = 'mumps',preconditioner = None, initial_relaxation = 0.5):
    # PETScOptions.set('ksp_view')
    # PETScOptions.set('ksp_monitor_true_residual')
    # PETScOptions.set('pc_type', 'fieldsplit')
    # PETScOptions.set('pc_fieldsplit_type', 'additive')
    # #PETScOptions.set('pc_fieldsplit_detect_saddle_point')
    # PETScOptions.set('fieldsplit_0_ksp_type', 'preonly')
    # PETScOptions.set('fieldsplit_0_pc_type', 'lu')
    # PETScOptions.set('fieldsplit_1_ksp_type', 'preonly')
    # PETScOptions.set('fieldsplit_1_pc_type', 'lu')
    # PETScOptions.set('fieldsplit_2_ksp_type', 'preonly')
    # PETScOptions.set('fieldsplit_2_pc_type', 'lu')
    #from IPython import embed; embed()
    # solve(Variation == 0,func,bc,J=Jacobian, form_compiler_parameters=ffc_options, 
    #   solver_parameters={"newton_solver":{"linear_solver":"minres",
    #                                       "preconditioner":"amg",
    #                                         "relaxation_parameter":initial_relaxation,
    #                                         'maximum_iterations': 50, 
    #                                         "absolute_tolerance": 1.0e-3, 
    #                                         "relative_tolerance": 1.0e-3}}) #icc + minr es works sort of
    
    solve(Variation == 0,func,bc,J=Jacobian, form_compiler_parameters=ffc_options, 
      solver_parameters={"newton_solver":{"linear_solver":linear_solver,
                                          "preconditioner":preconditioner,
                                            "relaxation_parameter":initial_relaxation,
                                            'maximum_iterations': 50, 
                                            "absolute_tolerance": 1.0e-3, 
                                            "relative_tolerance": 1.0e-3}}) 
    solve(Variation == 0,func,bc,J=Jacobian, form_compiler_parameters=ffc_options, 
      solver_parameters={"newton_solver":{"linear_solver":linear_solver, 
                                          "preconditioner":preconditioner,
                                            "relaxation_parameter":1.0,
                                            'maximum_iterations': 50, 
                                            "absolute_tolerance": 1.0e-8, 
                                            "relative_tolerance": 1.0e-8}}) 

def objective(func,F_target,dx,d):
    u, c, p = split(func)
    Δu = (Identity(d)+nabla_grad(u)).T*(Identity(d)+nabla_grad(u)) 
    #Δu_t = nabla_grad(u_target).T*nabla_grad(u_target)
    return inner(Δu-F_target,Δu-F_target)*dx
  
def minimizer(maxIter,tolObj,tolGrad,alphaInit,J,PDE,forwardJacobian,adjointPDE,shapeGradient,displacement,control,State,adjoint_state,ffc_options,Lagrangian,targetAngle):
  
  beta = 0.5 
  residuals = np.empty((0, 4))
  iteration = 0
  obj = 1
  normGrad = 1
  alphaMin = 1e-6
  sigma = 1e-1 # Armijo line search parameter.
  
  while iteration < maxIter and obj>tolObj and normGrad > tolGrad:
      #print('Inverse Tensor Components: ', InvTensorComps.vector()[:])
      # Solve and export the forward PDE.
      #forwardProblem = NonlinearProblem(PDE,u,[])
      SolveNonLinearProblem(PDE, State, forwardJacobian,[],ffc_options)

      # Evaluate the objective.
      obj = assemble(J)
      #print('objective',obj)

      # Solve the adjoint PDE.
      #adjointLinearProblem = LinearProblem(ufl.lhs(adjointPDE),ufl.rhs(adjointPDE))
      #p = adjointLinearProblem.solve()
      solve(lhs(adjointPDE) == rhs(adjointPDE), adjoint_state, 
        solver_parameters={'linear_solver': 'mumps', 
                          'preconditioner': 'ilu'})
      #solve_NLproblem(adjointPDE,p,body)

      # Evaluate the shape derivative.
      shapeDerivative = derivative(Lagrangian, control)
      shapeGradient.vector()[:] = assemble(shapeDerivative)

      # Evaluate the squared norm of the shape gradient induced by the (regularized)
      # elasticity inner product.
      normShapeGradient2 = sum(shapeGradient.vector() * assemble(shapeDerivative))

      # Set the initial step size.
      alpha = alphaInit

      # Store the mesh associated with the current iterate, as well as its objective value.
      #referenceMeshCoordinates = mesh.coordinates().copy()

      # Begin Armijo line search.
      lineSearchSuccessful = False
      while (lineSearchSuccessful == False) and (alpha > alphaMin):
          
          try:
              # Assign the mesh displacement vector field.
              displacement.assign(- alpha * shapeGradient)

              # Update the mesh by adding the displacement to the reference mesh.
              #InvTensorComps.vector()[:] = InvTensorComps.vector()[:] + displacement.vector()[:]
              control.assign(control- alpha * shapeGradient)
              #control.vector()[:] = control.vector()[:]%(2.0*np.pi)
              #ALE.move(mesh, displacement)

              # Solve the forward PDE.
              #forwardProblem = NonlinearProblem(PDE,u,[])
              SolveNonLinearProblem(PDE, State, forwardJacobian,[],ffc_options)
              # solve_NLproblem(PDE,u,body)
              #solve(lhs(forwardPDE) == rhs(forwardPDE), u_)

                  
          except RuntimeError:
              print('Runtime error due to no convergence, reducing alpha')
              alpha = beta * alpha
          
          else:
              # Evaluate the objective associated with the trial mesh.
              trialObj = assemble(J)

              # Evaluate the Armijo condition and reduce the step size if necessary.
              if (trialObj <= obj - sigma * alpha * normShapeGradient2):
                  lineSearchSuccessful = True
              else:
                  alpha = beta * alpha
              

      # Occasionally display some information.
      if (iteration % 10) == 0:
          print('-------------------------------------------')
          print('ITER        OBJ  ||GRADIENT||      ALPHA')
          print('-------------------------------------------')
          error = assemble((control-targetAngle)**2*dx)
          info = np.asarray([iteration, obj, math.sqrt(normShapeGradient2),error])
          residuals = np.vstack([residuals, info])
      print('{iteration:4d}  {objective:9.2e}  {gradientNorm:12.2e}  {alpha:9.2e}'.format(iteration = iteration, objective = obj, gradientNorm = math.sqrt(normShapeGradient2), alpha = alpha))
      
      normGrad = math.sqrt(normShapeGradient2)
      # Reset the step size for the next iteration.
      #  alpha = min(alphaInit, alpha / beta)

      # Increment the iteration counter.
      iteration += 1
      return control.vector()[:], State.vector()[:],adjoint_state.vector()[:], residuals


def PC(func,testfunc,dx):
  u, c, p = split(func)
  du, dc, dp = split(testfunc)
  return inner(nabla_grad(u)+nabla_grad(u).T,nabla_grad(du)+nabla_grad(du).T)*dx + inner(c,dc)*dx + p*dp*dx

def NewtonSolver(u0,func,Jacobian,F,SolutionSpace,M,τ):
  error = 1
  tol = 1e-5
  i = 0
  max_iter = 100
  u_i = Function(SolutionSpace)
  u_i.vector()[:] = u0.vector()[:]

  J_i = replace(Jacobian,{func:u_i})
  
  update = Function(SolutionSpace)

  F_i = replace(F,{func:u_i})

  P = assemble(M)
  print("Assembled P")
  while error>tol and i < max_iter: 
    #solve(J_i == F_i, update, 
     #     solver_parameters={'linear_solver': 'mumps', 
      #                    'preconditioner': 'ilu'}) #Compute update from tangent problem J(u,v) = F(u)

    
    #A, b = assemble_system(J_i, F_i)
    #P, _ = assemble_system(M, F_i)
    A = assemble(J_i)
    b = assemble(F_i)
    print("Assembled A and c")
    

    solver = PETScKrylovSolver("gmres",)
    solver.parameters["maximum_iterations"] = 1000
    #from IPython import embed; embed()
    #solver.set_operators(A,P)
    solver.set_operators(A,A)
    print("Set operators done")
    solver.solve(u_i.vector(), b)
    print("Solved")
    u_i.vector()[:] = u_i.vector()[:] - τ*update.vector()[:] #Update Solution
    #u_i.assign(u_i - τ*update)
    #F_i = replace(F,{func:u_i})
    error = sum(assemble(F_i) * assemble(F_i))
    print(error)
    #
    #print(sum(update[:] * update[:]))
    

    i += 1
  
def IterativeNewtonSolver(u0,func,Jacobian,F,SolutionSpace,τ):
  error = 1
  tol = 1e-5
  i = 0
  max_iter = 100
  u_i = Function(SolutionSpace)
  u_i.vector()[:] = u0.vector()[:]

  dofs_u = SolutionSpace.sub(0).dofmap().dofs()
  dofs_c = SolutionSpace.sub(1).dofmap().dofs()
  dofs_p = SolutionSpace.sub(2).dofmap().dofs()

  dofs_lambda = list(dofs_c) + list(dofs_p)
  #dofs_lambda = list(dofs_p)


  is_u = PETSc.IS().createGeneral(dofs_u, comm=PETSc.COMM_WORLD)
  is_lambda = PETSc.IS().createGeneral(dofs_lambda, comm=PETSc.COMM_WORLD)


  J_i = replace(Jacobian,{func:u_i})
  
  update = Function(SolutionSpace)

  F_i = replace(F,{func:u_i})

  #P = assemble(M)
  #print("Assembled P")
  while error>tol and i < max_iter:
    print(i) 
    #solve(J_i == F_i, update, 
     #     solver_parameters={'linear_solver': 'mumps', 
      #                    'preconditioner': 'ilu'}) #Compute update from tangent problem J(u,v) = F(u)

    
    #A, b = assemble_system(J_i, F_i)
    #P, _ = assemble_system(M, F_i)
    #A_fenics = assemble(J_i)
    #b_fenics = assemble(F_i)
    A_fenics, b_fenics = assemble_system(J_i, F_i)


    A = as_backend_type(A_fenics).mat()
    b = as_backend_type(b_fenics).vec()
    #print("Assembled A and c")
    

    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    #ksp.setType("gmres")
    ksp.setType(PETSc.KSP.Type.GMRES)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.FIELDSPLIT)
    #pc.setType("ilu")
    #pc.setType("fieldsplit")

    pc.setFieldSplitIS(
        ("u", is_u),
        ("lambda", is_lambda)
    )

    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)

    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

    # pc.setFromOptions()
    # pc.setUp()

    #ksp.setFromOptions()
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
    u_i.vector()[:] = u_i.vector()[:] - τ*update.vector()[:] #Update Solution
    #u_i.assign(u_i - τ*update)
    #F_i = replace(F,{func:u_i})
    error = sum(assemble(F_i) * assemble(F_i))
    print(error)
    #
    #print(sum(update[:] * update[:]))
    

    i += 1