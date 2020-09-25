import schnetpack as spk
import schnetpack.atomistic as atm
import schnetpack.representation as rep
from schnetpack.nn.cutoff import CosineCutoff
from schnetpack.data.loader import _collate_aseatoms
from schnetpack.environment import SimpleEnvironmentProvider

def make_schnet_model(args):
    label = args.label
    negative_dr=args.derivative is not None
    atomrefs = None
    #if hasattr(args,'atomrefs') and  args.atomrefs is not None:
    #    atomrefs = self.read_atomrefs(args.atomrefs,args.max_z)
    reps = rep.SchNet(n_atom_basis=args.num_filters, n_filters=args.num_filters, 
                    n_interactions=args.num_interactions, cutoff=args.cutoff, 
                    n_gaussians=args.num_gaussians, max_z=args.max_z, cutoff_network=CosineCutoff)
    output = spk.atomistic.Atomwise(n_in=reps.n_atom_basis, aggregation_mode='sum', 
                    property=label, derivative=args.derivative, negative_dr=negative_dr, 
                    mean=None, stddev=None, atomref=atomrefs)
    model = atm.AtomisticModel(reps, output)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters {}'.format(total_params))
    return model