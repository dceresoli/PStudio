PyObject* hartree(PyObject *self, PyObject *args)
{
    int l;
    PyArrayObject* nrdr_obj;
    PyArrayObject* r_obj;
    PyArrayObject* vr_obj;
    if (!PyArg_ParseTuple(args, "iOOO", &l, &nrdr_obj, &r_obj, &vr_obj))
        return NULL;

    const int M = PyArray_DIM(nrdr_obj, 0);
    const double* nrdr = DOUBLEP(nrdr_obj);
    const double* r = DOUBLEP(r_obj);
    double* vr = DOUBLEP(vr_obj);

    double p = 0.0;
    double q = 0.0;
    for (int g = M - 1; g > 0; g--)
    {
        double R = r[g];
        double rl = pow(R, l);
        double dp = nrdr[g] / rl;
        double rlp1 = rl * R;
        double dq = nrdr[g] * rlp1;
        vr[g] = (p + 0.5 * dp) * rlp1 - (q + 0.5 * dq) / rl;
        p += dp;
        q += dq;
    }
    vr[0] = 0.0;
    double f = 4.0 * M_PI / (2 * l + 1);
    for (int g = 1; g < M; g++)
    {
        double R = r[g];
        vr[g] = f * (vr[g] + q / pow(R, l));
    }
    Py_RETURN_NONE;
}
