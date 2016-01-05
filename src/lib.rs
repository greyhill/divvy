/// Given a vector of transfer times x and a vector of compute times c, 
/// compute the optimal work distribution t^* by 
///
/// t^* = argmin_{t >= 0, 1't = 1} || x + c .* t ||_\infty
///
/// i.e., find the work distribution such that all the nodes involved will
/// nominally finish at the same time.  This algorithm solves the above
/// linear program using the alternating directions methods of multipliers (ADMM).
#[allow(dead_code, unused_variables, unreachable_code, unused_mut)]
pub fn divide_work_linear(transfer_times: &[f32], compute_times: &[f32]) -> Vec<f32> {
    assert_eq!(transfer_times.len(), compute_times.len());
    let num_nodes = transfer_times.len();
    unimplemented!();

    let mut t = vec![1f32 / num_nodes as f32; num_nodes];
    let mut w = t.clone();
    let mut e = vec![0f32; num_nodes];

    let mut old_cost = cost(&t[..], &transfer_times[..], &compute_times[..]);
    loop {



        let new_cost = cost(&t[..], &transfer_times[..], &compute_times[..]);
        if (new_cost - old_cost).abs() < new_cost / 1000000f32 {
            break;
        }
        old_cost = new_cost;
    }
    t
}


/// Given a vector of transfer times x and a vector of compute times c, compute
/// an approximately optimal work distribution t^* by solving the smoothed
/// problem
///
/// t^* = argmin_{t >= 0, 1't = 1} || x + c .* t ||_p^p
///
/// with p > 2 even.  
pub fn divide_work_smooth(transfer_times: &[f32], compute_times: &[f32], p: u32) -> Vec<f32> {
    // check invariants
    assert_eq!(transfer_times.len(), compute_times.len());
    assert!(p >= 2u32);
    assert!(p % 2 == 0);

    // (diagonal) majorizer for cost
    let denom: Vec<f32> = transfer_times.iter().zip(compute_times.iter()).map(|(&xi, &ci)| {
        (p as f32 - 1f32)*ci.powf(2f32)*(xi + ci).powf(p as f32 - 2f32)
    }).collect();
    let mut t = vec![1f32 / transfer_times.len() as f32; transfer_times.len()];

    // TODO: check cost function for termination
    for _ in 0 .. 1024 {
        // compute gradient
        let g: Vec<f32> = transfer_times.iter().zip(compute_times.iter().zip(t.iter())).map(|(&xi, (&ci, &ti))| {
            ci * (xi + ci*ti).powf(p as f32 - 1f32)
        }).collect();

        // project gradient onto simplex tangent space
        let gn: Vec<f32> = {
            let gs = g.iter().fold(0f32, |l, &r| l + r);
            g.iter().map(|&gi| gi - gs).collect()
        };

        let mut step = 1f32;
        loop {
            // compute candidate update
            let tn: Vec<f32> = t.iter().zip(gn.iter().zip(denom.iter())).map(|(&ti, (&gni, &di))| ti - step*gni/di).collect();

            // check that candidate is nonnegative (feasible)
            let feasible = {
                let mut tr = true;
                for &ti in tn.iter() {
                    if ti < 0f32 {
                        tr = false;
                        break;
                    }
                }
                tr
            };

            // if not feasible, backtrack.  otherwise, progress to the next iteration
            if !feasible {
                step /= 2f32;
            } else {
                t = tn;
                break;
            }
        }

        // scale up to the simplex (suboptimal, but we're just trying to correct numerical errors)
        let ts = t.iter().fold(0f32, |l, &r| l+r);
        for mut ti in t.iter_mut() {
            *ti /= ts;
        }
    }

    t
}

fn cost(t: &[f32], x: &[f32], c: &[f32]) -> f32 {
    let mut max_val = 0f32;
    for (ti, (xi, ci)) in t.iter().zip(x.iter().zip(c.iter())) {
        let val = (xi + ci * ti).abs();
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

/// project z onto the l1 ball (simplex)
/// 
/// This implementation uses ADMM.
#[allow(dead_code, unused_variables, unreachable_code, unused_mut)]
fn proj_l1(z: &[f32]) -> Vec<f32> {
    unimplemented!()
}

#[test]
fn test_even() {
    let x = vec![0f32, 0f32, 0f32, 0f32];
    let c = vec![1f32, 1f32, 1f32, 1f32];
    let t = divide_work(&x[..], &c[..]);
    assert_eq!(t, vec![0.25, 0.25, 0.25, 0.25]);
}
