function [x,k,diff] = CG_algorithm(A, b, tol, max_iter)
    N=size(A,1);
    x0 = zeros(N, 1);
    r0 = b-A*x0;
    p0 = r0;
    
    x = x0;
    
    for j = 1:max_iter
        diff = norm(r0);
        k = j;
        if diff<tol
            break
        end
        alpha = dot(r0,r0)/dot(A*p0,p0);
        x = x + alpha*p0;
        r1 = r0 - alpha*A*p0;
        beta = dot(r1,r1)/dot(r0,r0);
        p0 = r1 + beta*p0;
        r0 = r1;
    end
end