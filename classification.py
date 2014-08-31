import numpy as np 
from random import uniform, shuffle
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from tabulate import tabulate

class Classification:
    def __init__(self, n):
        self.boundary = self.decision_boundary()
        self.x_in, self.y_in = self.sample_pts(n)
        self.linear_reg()
        self.nonlinear_reg()
        self.perceptron()
        self.svm()
        self.logistic()
        
    def decision_boundary(self):
    	#chooses a random line as the boundary
        pt1 = [uniform(-1,1), uniform(-1,1)]
        pt2 = [uniform(-1,1), uniform(-1,1)]
        boundary = np.array([pt1[1]*pt2[0]-pt1[0]*pt2[1], pt2[1]-pt1[1], pt1[0]-pt2[0]])
        return boundary

    def sample_pts(self, n):
    	x_in=[]
    	y_in=[]
        for i in range(n):
            x = np.array([1, uniform(-1,1), uniform(-1,1)])
            y = np.sign(np.dot(self.boundary, x))
            x_in.append(x)
            y_in.append(y)
        if len(set(y_in)) == 1: #checks if every point is on one side of the decision boundary 
            x_in, y_in = self.sample_pts(n)
        return np.array(x_in), np.array(y_in)

    def linear_reg(self):
        x_in, y_in = self.x_in, self.y_in
        w_linear = np.linalg.inv(x_in.T.dot(x_in)).dot(x_in.T).dot(y_in) 
        self.w_linear = w_linear

    def nonlinear_reg(self):
        x_in, y_in = self.x_in, self.y_in
        x2_in = np.array([[x[0],x[1],x[2],x[1]*x[2],x[1]**2,x[2]**2] for x in x_in])
        w_nonlinear = np.linalg.inv(x2_in.T.dot(x2_in)).dot(x2_in.T).dot(y_in)
        self.w_nonlinear = w_nonlinear

    def perceptron(self):
        x_in, y_in = self.x_in, self.y_in
        w_perceptron = np.array([0.0,0.0,0.0])
        restart = True
        order = [x for x in range(n_in)]
        shuffle(order)
        iterations = 0
        while restart:
            restart = False
            for i in range(n_in):        
                if np.sign(w_perceptron.T.dot(x_in[order[i]])) != y_in[order[i]]:
                    iterations += 1   
                    w_perceptron += y_in[order[i]]*x_in[order[i]]
                    restart = True
                    shuffle(order)
                    break

        self.iterations_perceptron = iterations
        self.w_perceptron = w_perceptron

    def logistic(self, learning_rate = 0.01):
        x_in, y_in = self.x_in, self.y_in
        w_logistic = np.array([0.0,0.0,0.0])
        error = 100
        while error > 0.01:
            order = [x for x in range(n_in)]
            shuffle(order)
            w_old = w_logistic
            for i in range(n_in):
                gradient = -y_in[order[i]] * x_in[order[i]]/(1 + np.exp(y_in[order[i]] * w_logistic.T.dot(x_in[order[i]])))
                w_logistic = w_logistic - learning_rate * gradient
            error = np.linalg.norm(w_old - w_logistic)
        self.w_logistic = w_logistic    
    
    def svm(self):
        x_in, y_in = self.x_in, self.y_in
        K = np.zeros((n_in,n_in))
        for i in range(n_in):
            for j in range(n_in):
                K[i,j] = x_in[i][-2:].T.dot(x_in[j][-2:])
        P = matrix(np.outer(y_in,y_in) * K)
        q = matrix(np.ones(n_in) * -1)
        A = matrix(y_in, (1,n_in))
        b = matrix(0.0)
        G = matrix(np.diag(np.ones(n_in)*-1))
        h = matrix(np.zeros(n_in))
        solvers.options['show_progress'] = False
        solution = solvers.qp(P,q,G,h,A,b)
        a = np.ravel(solution['x'])
        a = [a[i] if a[i] > 1e-5 else 0 for i in range(n_in)]
        w_svm = np.sum([a[i]*y_in[i]*x_in[i][-2:] for i in range(n_in)], axis=0)
        ind = a.index([x for x in a if x!=0][0]) #finds the index corresponding to support vector
        b = (y_in[ind] - w_svm.T.dot(x_in[ind][-2:]))
        w_svm = np.array([b, w_svm[0], w_svm[1]])
        self.w_svm = w_svm

    def performance_in(self):
    #measures performance of in sample
        x_in, y_in = self.x_in, self.y_in
        x2_in = np.array([[x[0],x[1],x[2],x[1]*x[2],x[1]**2,x[2]**2] for x in x_in])
        w_l = self.w_linear
        w_nl = self.w_nonlinear
        w_p = self.w_perceptron
        w_lg = self.w_logistic
        w_svm = self.w_svm
        error_l = 0
        error_nl = 0
        error_p = 0
        error_lg = 0
        error_svm = 0
        for i in range(n_in):
            if int(np.sign(w_l.T.dot(x_in[i]))) != y_in[i]:
                error_l += 1
            if int(np.sign(w_nl.T.dot(x2_in[i]))) != y_in[i]:
                error_nl += 1
            if int(np.sign(w_p.T.dot(x_in[i]))) != y_in[i]:
                error_p += 1
            if int(np.sign(w_lg.T.dot(x_in[i]))) != y_in[i]:
                error_lg += 1
            if int(np.sign(w_svm.T.dot(x_in[i]))) != y_in[i]:
                error_svm += 1
        return [error_l/float(n_in), error_nl/float(n_in), error_p/float(n_in), error_lg/float(n_in), error_svm/float(n_in)]

    def performance_out(self, n_out, iterations):
    # measures performance of out of sample testing
        errors_l = []
        errors_nl = []
        errors_p = []
        errors_lg = []
        errors_svm = []
        w_l = self.w_linear
        w_nl = self.w_nonlinear
        w_p = self.w_perceptron
        w_lg = self.w_logistic
        w_svm = self.w_svm

        for i in range(iterations):    
            x_out, y_out = self.sample_pts(n_out)
            x2_out = np.array([[x[0],x[1],x[2],x[1]*x[2],x[1]**2,x[2]**2] for x in x_out])
            error_l = 0
            error_nl = 0
            error_p = 0
            error_lg = 0
            error_svm = 0
            for j in range(n_out):
                if int(np.sign(w_l.T.dot(x_out[j]))) != y_out[j]:
                    error_l += 1
                if int(np.sign(w_nl.T.dot(x2_out[j]))) != y_out[j]:
                    error_nl += 1
                if int(np.sign(w_p.T.dot(x_out[j]))) != y_out[j]:
                    error_p += 1
                if int(np.sign(w_lg.T.dot(x_out[j]))) != y_out[j]:
                    error_lg += 1
                if int(np.sign(w_svm.T.dot(x_out[j]))) != y_out[j]:
                    error_svm += 1
            errors_l.append(error_l/ float(n_out))
            errors_nl.append(error_nl/float(n_out))
            errors_p.append(error_p/float(n_out))
            errors_lg.append(error_lg/float(n_out))
            errors_svm.append(error_svm/float(n_out))
        return [np.mean(errors_l), np.mean(errors_nl), np.mean(errors_p), np.mean(errors_lg), np.mean(errors_svm)]

    def plt(self):
        np.seterr(invalid='ignore')
        plt.axis([-1,1,-1,1])
        boundary = self.boundary
        x_in, y_in = self.x_in, self.y_in 
        w_l = self.w_linear
        w_nl = self.w_nonlinear
        w_p = self.w_perceptron
        w_lg = self.w_logistic
        w_svm = self.w_svm

        x = np.linspace(-1,1,100)
        draw_boundary = plt.plot(x, -boundary[1]/boundary[2]*x-boundary[0]/boundary[2], color='Yellow', linewidth=3, label='decision_boundary')

        for i in range(len(self.x_in)):
            plt.plot(x_in[i][1], x_in[i][2], color='r', marker='o' if y_in[i]>0 else 'x')

        draw_linear = plt.plot(x, -w_l[1]/w_l[2]*x-w_l[0]/w_l[2], color='DodgerBlue', linewidth=1, label='linear_regression')
        draw_nonlinear = plt.plot(x, np.sqrt(((w_nl[2]+w_nl[3]*x)/(2*w_nl[5]))**2-w_nl[0]/w_nl[5]-w_nl[1]*x/[5]-(w_nl[4]*(x**2)/w_nl[5]))-(w_nl[2]+w_nl[3]*x)/(2*w_nl[5]), color ='Pink', linewidth=1, label='nonlinear_transformation')
        draw_nonlinear = plt.plot(x, -1*np.sqrt(((w_nl[2]+w_nl[3]*x)/(2*w_nl[5]))**2-w_nl[0]/w_nl[5]-w_nl[1]*x/[5]-(w_nl[4]*(x**2)/w_nl[5]))-(w_nl[2]+w_nl[3]*x)/(2*w_nl[5]), color ='Pink', linewidth=1)
        draw_perceptron= plt.plot(x, -w_p[1]/w_p[2]*x-w_p[0]/w_p[2], color='Green', linewidth=1, label='perceptron')
        draw_logistic = plt.plot(x, -w_lg[1]/w_lg[2]*x-w_lg[0]/w_lg[2], color='Orange', linewidth=1, label='logistic regression')
        draw_svm = plt.plot(x, -w_svm[1]/w_svm[2]*x-w_svm[0]/w_svm[2], color='Purple', linewidth=1, label='support vector machines')
        plt.legend(loc='best', prop={'size':10})
        plt.show()

if __name__ == "__main__":
    n_in = 100
    n_out = 100
    iterations = 100
    error = []

    for i in range(iterations):
        c = Classification(n_in)
        error.append(c.performance_in())
    e_in = [sum(x)/float(iterations) for x in zip(*error)]

    c = Classification(n_in)
    e_out = c.performance_out(n_out,iterations)

    rows = ["linear regression", "nonlinear regression", "perceptron algorithm", "logistic regression", "support vector machines"]
    result = zip(rows, e_in, e_out)
    print tabulate(result, headers=["Method", "in-sample error","out-of-sample error"])
    c.plt()


