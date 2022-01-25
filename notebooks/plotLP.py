"""
plotLP.py: Helper functions to visualize the first two dimensions of an LP.
Author: Tian You
"""

import numpy as np
import pulp
import matplotlib.pyplot as plt

def plotLP(LP, x_lower, x_upper, y_lower, y_upper,
           draw_obj=True, show_feasible=True, int_prog=False, 
           grid_pts=200, fig_size=(6,6)) -> None:
    
    """Generate a plot of the first two variables for the given linear program LP.

    Args:
      LP: An LP object from PuLP.
      x_lower: A float representing the lowest value of x shown in the graph.
      x_upper: A float representing the highest value of x shown in the graph.
      y_lower: A float representing the lowest value of y shown in the graph.
      y_upper: A float representing the highest value of y shown in the graph.
      draw_obj: A boolean indicating whether to draw the objective function. 
      show_feasible: A boolean indicating whether to show the feasible region.
      int_prog: A boolean indicating whether to treat the LP as an integer LP 
          when showing the feasible region. Default to False.
      grid_pts: An integer representing the number of points in for the xy-grid.
      fig_size: A tuple of integers specifying the size of the plot.

    Returns:
      None.
    """
    
    # initialize the plot
    fig, ax = plt.subplots(figsize=fig_size)
    
    # extract the first two variables from LP
    var1 = LP._variables[0]; var2 = LP._variables[1]

    # set up the grid
    x_line = np.linspace(x_lower, x_upper, num=grid_pts) 
    y_line = np.linspace(y_lower, y_upper, num=grid_pts)
    if int_prog:
        x_int = np.arange(x_lower, x_upper)
        y_int = np.arange(y_lower, y_upper)
        x_grid, y_grid = np.meshgrid(x_int, y_int)
    else:
        x_grid, y_grid = np.meshgrid(x_line, y_line)
    
    # initialize the feasible region
    feasible_region = (x_grid==x_grid)
     
    # draw the constraints
    constr_keys = LP.constraints.keys()    
    for constr_key in constr_keys:
        constr = LP.constraints[constr_key]
        label = str(constr)
        const = -1*constr.constant
        coef1 = constr.get(var1)
        coef2 = constr.get(var2)

        if coef1 == None:  # e.g. x2 >= 2
            coef1 = 0
        if coef2 == None:  # e.g. x1 <= 5
            coef2 = 0
            y = y_line
            x = (const-0*y)/coef1
        else:
            x = x_line
            y = (const-coef1*x)/coef2
        plt.plot(x, y, label=label)

        # compute the feasible region
        if show_feasible:
            lhs = coef1 * x_grid + coef2 * y_grid
            rhs = const
            sense = constr.sense # sign of the inequality
            if sense == -1:
                new_region = (lhs <= rhs)
            elif sense == 1:
                new_region = (lhs >= rhs)
            elif sense == 0:
                new_region = (lhs == rhs)
            feasible_region = feasible_region & new_region

    # draw the objective function
    if draw_obj:
        obj = LP.objective
        label = "Obj: " + str(obj)
        coef1 = obj.get(var1)
        coef2 = obj.get(var2)
        if coef1 == None:    # e.g min x2
            plt.hlines(y=y_upper*0.8, xmin=x_lower, xmax=x_upper, label=label, 
                       linewidth=3, color='indigo', linestyle='--')
        elif coef2 == None:  # e.g. max 3*x1
            plt.vlines(x=x_upper*0.8, ymin=y_lower, ymax=y_upper, label=label, 
                       linewidth=3, color='indigo', linestyle='--')
        else:
            x = x_line
            y = y_upper*0.8-coef1*x/coef2
            plt.plot(x, y, label=label, linewidth=3, color='indigo', linestyle='--')
            
    # show the feasible region
    if show_feasible:
        var1.Ub = var1.getUb()
        var1.Lb = var1.getLb()
        if var1.Ub != None:
            feasible_region = feasible_region & (x_grid <= var1.Ub * np.ones_like(x_grid))
        if var1.Lb != None:
            feasible_region = feasible_region & (x_grid >= var1.Lb * np.ones_like(x_grid))
        
        var2.Ub = var2.getUb()
        var2.Lb = var2.getLb()
        if var2.Ub != None:
            feasible_region = feasible_region & (y_grid <= var1.Ub * np.ones_like(y_grid))
        if var2.Lb != None:
            feasible_region = feasible_region & (y_grid >= var1.Lb * np.ones_like(y_grid))            
        
        if int_prog:  # draw grey dots for IP
            x_grid = x_grid.flatten()[feasible_region.flatten() == True]
            y_grid = y_grid.flatten()[feasible_region.flatten() == True]
            plt.plot(x_grid, y_grid, 'o', color = 'grey')
        else:         # draw grey areas for LP
            plt.imshow(feasible_region.astype(int), 
                       extent=(x_grid.min(),x_grid.max(),y_grid.min(),y_grid.max()),origin="lower",
                       cmap="Greys", alpha = 0.3)
    
    # format the plot
    ax.spines['left'].set_position(('data', 0))            # y axis crossing at 0
    ax.spines['bottom'].set_position(('data', 0))          # x axis crossing at 0
    ax.spines['top'].set_visible(False)                    # hide top border
    ax.spines['right'].set_visible(False)                  # hide right border
    plt.xlim(x_lower, x_upper); plt.ylim(y_lower, y_upper) # set axis limit
    plt.xlabel(str(var1)); plt.ylabel(str(var2))           # assign axis labels 
    plt.legend(loc='upper right')                          # add the legend
    

    
def read_vec(v) -> int:
    
    """Determine if the given vector v is the column corresponding to a basic variable.
       If it is not a column of a basic variable, return -1.
       If it is a column of a basic variable, return the row index of the solution.

    Args:
       v: an ndarray of float corresponds to the column of a variable.

    Returns:
       -1: if v is not a column of a basic variable.
       int >= 0: a positive integer representing the row index of the solution 
          if v is a column of a basic variable.
    """
    
    sol = False
    for i in np.arange(0, len(v)):
        curr = v[i]
        if sol == True:
            if curr != 0:
                sol = False
                return -1
        else:
            if curr == 1:
                sol = True
                row = i
            elif curr != 0:
                sol = False
                return -1
    if sol:
        return row
    else:
        return -1

    

def xy_coord(T, x_col_num=0, y_col_num=1):
    
    """Return the coordinates of the first two variables x and y given the tableau T.

    Args:
       T: An ndarray (matrix) representing the tabeau.
       x_col_num: An integer representing the column index of x in the tableaus. Default to 0.
       y_col_num: An integer representing the column index of y in the tableaus. Default to 1.

    Returns:
       A tuple of two lists of length 1 reprensenting the coordinates of x and y.
       Lists are used to make the plotting step easier.
    """
    
    x_col = T[:,x_col_num]
    y_col = T[:,y_col_num]    
    x_row = read_vec(x_col)
    y_row = read_vec(y_col)
    x_value = 0.0 if x_row == -1 else T[x_row,-1]
    y_value = 0.0 if y_row == -1 else T[y_row,-1]
    return ([x_value], [y_value])


def plotVert(curr_T, past_T=None, x_col_num=0, y_col_num=1) -> None:
    
    """Plot the vertices corresponding to the given tableaus.

    Args:
       curr_T: An ndarray (matrix) representing the current tableau.
       past_T: A list of ndarrays (matrices) representing the tableaus in the past.
       x_col_num: An integer representing the column index of x in the tableaus. Default to 0.
       y_col_num: An integer representing the column index of y in the tableaus. Default to 1.

    Returns:
       None.
    """
        
    past_coords = [[],[]]
    if past_T != None:
        for T in past_T:
            past_coord = xy_coord(T, x_col_num, y_col_num)
            past_coords[0] += past_coord[0]
            past_coords[1] += past_coord[1]
    plt.plot(past_coords[0],past_coords[1],'o', color='blue')
    
    curr_coord = xy_coord(curr_T, x_col_num, y_col_num)
    plt.plot(curr_coord[0],curr_coord[1],'o', color='crimson')
    