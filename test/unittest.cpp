#include "unittest.hpp"


TEST(FastUpdate, BlockMatrixAdd)
{
  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  std::vector<size_t> N_list, M_list;
  N_list.push_back(0);
  N_list.push_back(10);
  N_list.push_back(2);

  M_list.push_back(10);
  M_list.push_back(20);
  M_list.push_back(4);

  for (size_t n=0; n<N_list.size(); ++n) {
    for (size_t m=0; m<M_list.size(); ++m) {
      const size_t N = N_list[n];
      const size_t M = M_list[m];

      matrix_t A(N,N), B(N,M), C(M,N), D(M,M);
      matrix_t E(N,N), F(N,M), G(M,N), H(M,M);
      alps::ResizableMatrix<Scalar> invA(N,N), BigMatrix(N+M, N+M, 0);//, invBigMatrix2(N+M, N+M, 0);

      randomize_matrix(A, 100);//100 is a seed
      randomize_matrix(B, 200);
      randomize_matrix(C, 300);
      randomize_matrix(D, 400);
      if (N>0) {
        invA = A.inverse();
      } else {
        invA.destructive_resize(0,0);
      }

      copy_block(A,0,0,BigMatrix,0,0,N,N);
      copy_block(B,0,0,BigMatrix,0,N,N,M);
      copy_block(C,0,0,BigMatrix,N,0,M,N);
      copy_block(D,0,0,BigMatrix,N,N,M,M);

      const Scalar det_rat = compute_det_ratio_up<Scalar>(B, C, D, invA);
      ASSERT_TRUE(std::abs(det_rat-determinant(BigMatrix)/A.determinant())<1E-8)
                << "N=" << N << " M=" << M << " " << std::abs(det_rat-determinant(BigMatrix)) << "/" << std::abs(det_rat)<<"="
                << std::abs(det_rat-determinant(BigMatrix)/A.determinant());

      const Scalar det_rat2 = compute_inverse_matrix_up(B, C, D, invA);
      ASSERT_TRUE(std::abs(det_rat-det_rat2)<1E-8) << "N=" << N << " M=" << M;
      ASSERT_TRUE(norm_square(inverse(BigMatrix)-invA)<1E-8) << "N=" << N << " M=" << M;
    }
  }
}


TEST(FastUpdate, BlockMatrixRemove)
{
  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

  std::vector<int> N_list, M_list;
  N_list.push_back(10);
  M_list.push_back(10);
  M_list.push_back(20);
  M_list.push_back(30);

  for (int n=0; n<N_list.size(); ++n) {
    for (int m=0; m<M_list.size(); ++m) {
      const int N = N_list[n];
      const int M = M_list[m];

      matrix_t BigMatrix(N+M, N+M, 0), invBigMatrix(N+M, N+M, 0);//G, G^{-1}
      matrix_t SmallMatrix(N,N,0);//G'
      std::vector<std::pair<int,int> > swapped_rows_in_G, swapped_cols_in_G;

      randomize_matrix(BigMatrix, 100);//100 is a seed
      invBigMatrix = inverse(BigMatrix);

      //which rows and cols are to be removed
      std::vector<int> rows_removed(N+M);
      std::vector<int> rows_remain(N);
      for (int i=0; i<N+M; ++i) {
        rows_removed[i] = i;
      }
      std::random_shuffle(rows_removed.begin(), rows_removed.end());
      for (int i=0; i<N; ++i) {
        rows_remain[i] = rows_removed[i+M];
      }
      rows_removed.resize(M);
      std::sort(rows_removed.begin(), rows_removed.end());
      std::sort(rows_remain.begin(), rows_remain.end());

      for (int j=0; j<N; ++j) {
        for (int i=0; i<N; ++i) {
          SmallMatrix(i,j) = BigMatrix(rows_remain[i], rows_remain[j]);
        }
      }

      //testing compute_det_ratio_down
      double det_rat = compute_det_ratio_down(M,rows_removed,invBigMatrix);
      ASSERT_TRUE(std::abs(det_rat-alps::numeric::determinant(SmallMatrix)/alps::numeric::determinant(BigMatrix))<1E-8) << "N=" << N << " M=" << M;

      matrix_t invSmallMatrix2(invBigMatrix);
      double det_rat2 = compute_inverse_matrix_down(M,rows_removed,invSmallMatrix2, swap_list);
      ASSERT_TRUE(std::abs(det_rat-det_rat2)<1E-8) << "N=" << N << " M=" << M;

      matrix_t SmallMatrix3(BigMatrix);
      for (int s=0; s<swap_list.size(); ++s) {
        SmallMatrix3.swap_cols(swap_list[s].first, swap_list[s].second);
        SmallMatrix3.swap_rows(swap_list[s].first, swap_list[s].second);
      }
      SmallMatrix3.resize(N,N);
      ASSERT_TRUE(alps::numeric::norm_square(inverse(SmallMatrix3)-invSmallMatrix2)<1E-8) << "N=" << N << " M=" << M;
    }
  }
}

/*
TEST(FastUpdate, ReplaceRow) {
    typedef double T;
    const int N = 100;
    const int m = 41;
    alps::numeric::matrix<T> D(N, N), new_D(N,N), M(N, N), new_M, new_M_fastu, Dr(1,N);

    randomize_matrix(D, 100);//100 is a seed
    M = inverse(D);

    randomize_matrix(Dr, 200);
    new_D = D;
    for (int i=0; i<N; ++i) {
        new_D(m,i) = Dr(0,i);
    }
    new_M = inverse(new_D);

    double det_rat = determinant(new_D)/determinant(D);
    double det_rat_fastu = compute_det_ratio_replace_row(M,Dr,m);
    ASSERT_TRUE(std::abs(det_rat/det_rat_fastu-1)<1E-8);

    new_M_fastu = M;
    compute_imverse_matrix_replace_row(new_M_fastu,Dr,m);
    ASSERT_TRUE(alps::numeric::norm_square(new_M-new_M_fastu)<1E-8);
}

TEST(FastUpdate, ReplaceRowCol) {
    typedef double T;
    const int N = 10;
    const int m = 3;
    assert(m<N);
    alps::numeric::matrix<T> D(N, N), new_D(N,N), M(N, N), new_M, new_M_fastu, Dr(1,N), Dc(N,1);

    randomize_matrix(D, 100);//100 is a seed
    randomize_matrix(Dr, 200);
    randomize_matrix(Dc, 300);

    M = inverse(D);
    Dr(0,m) = Dc(m,0);
    new_D = D;
    for (int i=0; i<N; ++i) {
        new_D(m,i) = Dr(0,i);
        new_D(i,m) = Dc(i,0);
    }
    new_M = inverse(new_D);

    double det_rat = determinant(new_D)/determinant(D);
    alps::numeric::matrix<T> M_fastu(M);
    double det_rat_fastu = compute_inverse_matrix_replace_row_col(M_fastu,Dr,Dc,m);
    ASSERT_TRUE(std::abs(det_rat/det_rat_fastu-1)<1E-10);

    ASSERT_TRUE(alps::numeric::norm_square(new_M-M_fastu)<1E-10);
}

TEST(FastUpdate, BlockMatrixReplaceRowsColsSuccessive) {
    typedef alps::numeric::matrix<double> matrix_t;

    std::vector<size_t> N_list, M_list;
    N_list.push_back(10);
    M_list.push_back(4);

    for (int n = 0; n < N_list.size(); ++n) {
        for (int m = 0; m < M_list.size(); ++m) {
            const int N = N_list[n];
            const int M = M_list[m];
            matrix_t BigMatrix(N + M, N + M, 0), invBigMatrix(N + M, N + M, 0);

            //which rows and cols are to be replaced
            std::vector<int> rows_replaced(N + M);
            for (int i = 0; i < N + M; ++i) {
                rows_replaced[i] = i;
            }
            std::random_shuffle(rows_replaced.begin(), rows_replaced.end());
            rows_replaced.resize(M);
            std::sort(rows_replaced.begin(), rows_replaced.end());

            randomize_matrix(BigMatrix, 100);//100 is a seed

            matrix_t R(M, N, 0), S(M, M, 0), Q(N, M, 0);
            randomize_matrix(R, 110);//100 is a seed
            randomize_matrix(Q, 310);//100 is a seed
            randomize_matrix(S, 210);//100 is a seed

            matrix_t BigMatrixReplaced(BigMatrix);
            replace_rows_cols_respect_ordering(BigMatrixReplaced, Q, R, S, rows_replaced);

            //testing compute_det_ratio_down
            invBigMatrix = inverse(BigMatrix);
            double det_rat = alps::numeric::determinant(BigMatrixReplaced)/determinant(BigMatrix);

            double det_rat_fast = compute_inverse_matrix_replace_rows_cols_succesive(invBigMatrix,Q,R,S,rows_replaced);
            ASSERT_TRUE(std::abs(det_rat-det_rat_fast)<1E-8);
        }
    }
}

TEST(FastUpdate, BlockMatrixReplaceRowsCols) {
    std::vector<size_t> N_list, M_list;
    N_list.push_back(10);
    M_list.push_back(4);

    N_list.push_back(100);
    M_list.push_back(50);

    N_list.push_back(100);
    M_list.push_back(49);

    for (int n = 0; n < N_list.size(); ++n) {
        for (int m = 0; m < M_list.size(); ++m) {
            const int N = N_list[n];
            const int M = M_list[m];

            typedef alps::numeric::matrix<double> matrix_t;

            matrix_t BigMatrix(N + M, N + M, 0), invBigMatrix(N + M, N + M, 0);
            std::vector<std::pair<int, int> > swap_list;

            randomize_matrix(BigMatrix, 100);//100 is a seed
            invBigMatrix = inverse(BigMatrix);

            //which rows and cols are to be replaced
            std::vector<int> rows_replaced(N + M);
            for (int i = 0; i < N + M; ++i) {
                rows_replaced[i] = i;
            }
            std::random_shuffle(rows_replaced.begin(), rows_replaced.end());
            rows_replaced.resize(M);
            std::sort(rows_replaced.begin(), rows_replaced.end());

            swap_list.resize(M);
            for (int i=0; i<M; ++i) {
                swap_list[i] = std::pair<int,int>(rows_replaced[M-1-i], N+M-1-i);
            }

            matrix_t R(M, N), S(M, M), Q(N, M);
            randomize_matrix(R, 110);//100 is a seed
            randomize_matrix(S, 210);//100 is a seed
            randomize_matrix(Q, 310);//100 is a seed

            matrix_t BigMatrixReplaced(BigMatrix);
            replace_rows_cols(BigMatrixReplaced, Q, R, S, rows_replaced);

            //testing compute_det_ratio_down
            double det_rat = alps::numeric::determinant(BigMatrixReplaced)/determinant(BigMatrix);

            matrix_t invBigMatrix_fast(invBigMatrix), Mmat, inv_tSp, tPp, tQp, tRp, tSp;
            swap_cols_rows(invBigMatrix_fast, swap_list.begin(), swap_list.end());
            double det_rat_fast = compute_det_ratio_replace_rows_cols(invBigMatrix_fast, Q, R, S, Mmat, inv_tSp);
            //compute_inverse_matrix_replace_rows_cols(invBigMatrix_fast, Q, R, S, Mmat, inv_tSp, tPp, tQp, tRp, tSp);
            compute_inverse_matrix_replace_rows_cols(invBigMatrix_fast, Q, R, S, Mmat, inv_tSp);
            swap_cols_rows(invBigMatrix_fast, swap_list.rbegin(), swap_list.rend());
            ASSERT_TRUE(std::abs(det_rat-det_rat_fast)<1E-8);
            ASSERT_TRUE(alps::numeric::norm_square(inverse(BigMatrixReplaced)-invBigMatrix_fast)<1E-8);
        }
    }
}

TEST(FastUpdate, BlockMatrixReplaceLastRowsColsWithDifferentSizes) {
    typedef alps::numeric::matrix<double> matrix_t;

    std::vector<size_t> N_list, M_list, Mold_list;
    N_list.push_back(10);
    M_list.push_back(4);
    Mold_list.push_back(5);

    N_list.push_back(100);
    M_list.push_back(40);
    Mold_list.push_back(50);

    N_list.push_back(100);
    M_list.push_back(49);
    Mold_list.push_back(20);

    N_list.push_back(100);
    M_list.push_back(100);
    Mold_list.push_back(20);

    for (int n = 0; n < N_list.size(); ++n) {
        for (int m = 0; m < M_list.size(); ++m) {
            const int N = N_list[n];
            const int M = M_list[m];
            const int Mold = Mold_list[m];

            matrix_t BigMatrix(N + Mold, N + Mold, 0), invBigMatrix(N + Mold, N + Mold, 0);

            randomize_matrix(BigMatrix, 100);//100 is a seed
            invBigMatrix = inverse(BigMatrix);

            matrix_t R(M, N), S(M, M), Q(N, M);
            randomize_matrix(R, 110);//100 is a seed
            randomize_matrix(S, 210);//100 is a seed
            randomize_matrix(Q, 310);//100 is a seed

            matrix_t BigMatrixReplaced(N+M, N+M);
            alps::numeric::copy_block(BigMatrix, 0, 0, BigMatrixReplaced, 0, 0, N, N);
            alps::numeric::copy_block(Q, 0, 0, BigMatrixReplaced, 0, N, N, M);
            alps::numeric::copy_block(R, 0, 0, BigMatrixReplaced, N, 0, M, N);
            alps::numeric::copy_block(S, 0, 0, BigMatrixReplaced, N, N, M, M);

            //testing compute_det_ratio_down
            double det_rat = alps::numeric::determinant(BigMatrixReplaced)/determinant(BigMatrix);

            matrix_t invBigMatrix_fast(invBigMatrix), Mmat, inv_tSp, tPp, tQp, tRp, tSp;
            double det_rat_fast = compute_det_ratio_replace_rows_cols(invBigMatrix_fast, Q, R, S, Mmat, inv_tSp);
            compute_inverse_matrix_replace_rows_cols(invBigMatrix_fast, Q, R, S, Mmat, inv_tSp);
            ASSERT_TRUE(std::abs(det_rat-det_rat_fast)/std::abs(det_rat)<1E-8);
            ASSERT_TRUE(alps::numeric::norm_square(inverse(BigMatrixReplaced)-invBigMatrix_fast)<1E-8);
        }
    }
}

TEST(Boost, Binomial) {
    const size_t k = 2;
    for (size_t N=k; N<10; ++N) {
        const double tmp = boost::math::binomial_coefficient<double>(N,k);
        ASSERT_TRUE(std::abs(tmp-0.5*N*(N-1.0))<1E-8);
    }
}

TEST(MyUtil, permutation) {
    assert(permutation(3,1)==3);
    assert(permutation(3,2)==6);
}

TEST(QuantumNumber, diagonal_GF) {
    size_t n_site = 4;
    size_t n_flavors = 2;
    size_t N=1000;

    const size_t n_rank=2;
    const size_t n_af=2;
    const size_t Nv=1;
    const double eps=0;

    green_function<double> gf(N, n_site, n_flavors);
    for (int t=0; t<N; ++t) {
        for (int flavor=0; flavor<n_flavors; ++flavor) {
            for (int i=0; i<n_site; ++i) {
                for (int j=0; j<n_site; ++j) {
                    gf(t, i, j, flavor) = 0.0;
                }
                gf(t, i, i, flavor) = -0.5;
            }
        }
    }

    std::vector<vertex_definition<double> > vertices;
    boost::multi_array<double,2> alpha(boost::extents[n_af][n_rank]);
    std::fill(alpha.origin(), alpha.origin()+alpha.num_elements(), 0);

    //for (size_t iv=0; iv<Nv; ++iv) {
    std::vector<spin_t> flavors(n_rank);
    std::vector<size_t> sites(2*n_rank);
    flavors[0] = 0;
    flavors[1] = 1;
    sites[0] = 0;
    sites[1] = 1;
    sites[2] = 2;
    sites[3] = 3;
    vertices.push_back(vertex_definition<double>(2,2,flavors,sites,0.0,alpha,0));

    int qs[] = {1, -1, 0, 0, 0, 0, 1, -1};
    std::vector<std::vector<std::valarray<int> > > quantum_number_vertices;
    std::vector<std::vector<std::vector<size_t> > > groups(n_flavors);
    std::vector<std::vector<int> > group_map;
    quantum_number_vertices = make_quantum_numbers<double,double,green_function<double> >(gf, vertices, groups, group_map, eps);
    std::valarray<int> qs2(qs,n_site*n_flavors);
    bool flag = true;
    int i_af = 0;
    for (int i=0; i<qs2.size(); ++i) {
        if (qs2[i]!=quantum_number_vertices[0][i_af][i]) {
            flag = false;
        }
    }
    ASSERT_TRUE(flag);

    //vertices[0].make_quantum_numbers(group_map, quantum_number_vertices[0].size()/n_flavors);

    //ASSERT_TRUE(qs2==std::valarray<int>(quantum_number_vertices[0]));
    //ASSERT_TRUE(1==1);
    //ASSERT_TRUE(qs2==qs3);
}

TEST(UpdateStatistics, EstimateSpread) {
    const double beta = 100;

    std::vector<simple_vertex> vertices;
    vertices.push_back(simple_vertex(0.0));
    vertices.push_back(simple_vertex(0.3*beta));
    ASSERT_TRUE(std::abs(compute_spread<std::vector<simple_vertex> >(vertices.begin(), vertices.end(), beta)/beta-0.3)<1E-5);

    //vertices are distributed on [0,beta] uniformly.
    for (int Nv=2; Nv<10; ++Nv) {
        vertices.clear();
        for (int iv=0; iv<Nv; ++iv) {
            vertices.push_back(simple_vertex((beta*iv)/Nv));
        }
        ASSERT_TRUE(std::abs(compute_spread<std::vector<simple_vertex> >(vertices.begin(), vertices.end(), beta)/beta-(1-1./Nv))<1E-5);
    }
}

TEST(TypeRange, Integer) {
    ASSERT_TRUE(std::numeric_limits<int>::max()>=2147483647);
}

TEST(Util, Mymod) {
    const double tau = 1.0, beta = 15.0;
    for (int i=-10; i<=10; ++i) {
        double x = tau+beta*i;
        ASSERT_TRUE(std::abs(mymod(x,beta)-tau)<1E-10);
    }
}

TEST(FastUpdate, BlockMatrixReplaceRowsColsSingular) {
    typedef alps::numeric::matrix<double> matrix_t;

    for (int i=0; i<20; ++i) {
        double alpha = pow(0.1, 20*i);
        double a=4., b=8.;

        matrix_t G2(2,2), G2p(2,2), M2p(2,2);
        std::vector<std::pair<int, int> > swap_list;

        G2(0,0) = alpha;
        G2(1,1) = alpha;
        G2(1,0) = a;
        G2(0,1) = a;

        M2p(0,0) = alpha;
        M2p(1,0) = -b;
        M2p(0,1) = -b;
        M2p(1,1) = alpha;
        M2p /= (alpha+b)*(alpha-b);

        matrix_t Q(1,1), R(1,1), S(1,1);
        Q(0,0) = b;
        R(0,0) = b;
        S(0,0) = alpha;

        matrix_t M2p_fast = inverse(G2);
        matrix_t Mmat, inv_tSp;
        matrix_t tPp, tQp, tRp, tSp;

        double det_rat_fast = compute_det_ratio_replace_rows_cols_safe(M2p_fast, Q, R, S, Mmat, inv_tSp);
        double det_rat = (alpha-b)*(alpha+b)/((alpha-a)*(alpha+a));
    }
}

TEST(FastUpdate, ReplaceDiagonalElements) {
    typedef double T;
    typedef alps::numeric::matrix<T> matrix_t;

    const int N=10, m=2, offset=2;
    //const int N=2, m=1, offset=0;
    assert(m+offset<=N);

    matrix_t A_old(N,N), A_new(N,N), new_elems(m,1);
    std::vector<T> elems_diff(m);
    std::vector<int> pos(m);

    randomize_matrix(A_old, 100);
    randomize_matrix(new_elems, 200);
    A_new = A_old;
    matrix_t invA_old = inverse(A_old);
    for (int i=0; i<m; ++i) {
        pos[i] = i+offset;
    }
    for (int i=0; i<m; ++i) {
        elems_diff[i] = new_elems(i,0)-A_old(pos[i],pos[i]);
        A_new(pos[i],pos[i]) = new_elems(i,0);
    }

    const T det_rat = determinant(A_new)/determinant(A_old);
    const T det_rat_fast = compute_det_ratio_replace_diaognal_elements(invA_old, m, pos, elems_diff, true);
    ASSERT_TRUE(std::abs((det_rat-det_rat_fast)/det_rat)<1E-8);

    // inverse matrix update
    matrix_t invA_new = inverse(A_new);
    matrix_t invA_new_fast = invA_old;
    compute_det_ratio_replace_diaognal_elements(invA_new_fast, m, pos, elems_diff, false);

    ASSERT_TRUE(std::abs(alps::numeric::norm_square(invA_new-invA_new_fast))<1E-5);
}

TEST(MatrixLibrary, submatrix_view) {
    typedef double T;
    const size_t M = 50, N1 = 10, N2 = 20, start_row=5, start_col=10;
    alps::numeric::matrix<T> A(M,M, 1.0), B(M,M, 1.0), C_subcopy(M,M,1.0), C(2*M,2*M, 0.0), C_sub(N1,N2);

    randomize_matrix(A, 100);//100 is a seed
    randomize_matrix(B, 200);//200 is a seed

    A.resize(N1,N1);
    B.resize(N1,N2);
    C.resize(M,M);

    alps::numeric::submatrix_view<T> C_subview(C, start_row, start_col, N1, N2);
    mygemm(1.0, A, B, (T)0.0, C_subview);
    mygemm(1.0, A, B, (T)0.0, C_sub);

    for (int j=0; j<N2; ++j) {
        for (int i=0; i<N1; ++i) {
            ASSERT_TRUE(std::abs(C_sub(i,j)-C(i+start_row,j+start_col))/std::abs(C_sub(i,j))<1E-8);
            ASSERT_TRUE(std::abs(C_subview(i,j)-C(i+start_row,j+start_col))/std::abs(C_subview(i,j))<1E-8);
        }
    }

    //copy from a view to a matrix
    alps::numeric::my_copy_block(C_subview, 0, 0, C_subcopy, start_row, start_col, N1, N2);
    for (int j=0; j<N2; ++j) {
        for (int i = 0; i < N1; ++i) {
            ASSERT_TRUE(std::abs(C_subview(i,j)-C_subcopy(i+start_row,j+start_col))/std::abs(C_subview(i,j))<1E-8);
        }
    }

    //copy from a matrix to a view
    alps::numeric::my_copy_block(C_subcopy, start_row, start_col, C_subview, 0, 0, N1, N2);
    for (int j=0; j<N2; ++j) {
        for (int i = 0; i < N1; ++i) {
            ASSERT_TRUE(std::abs(C_subview(i,j)-C_subcopy(i+start_row,j+start_col))/std::abs(C_subview(i,j))<1E-8);
        }
    }
}

TEST(SubmatrixUpdate, single_vertex_insertion_spin_flip)
{
    typedef std::complex<double> T;
    const int n_sites = 3;
    const double U = 2.0;
    const double alpha = 1E-2;
    const double beta = 200.0;
    const int Nv_max = 2;
    const int n_flavors = 2;
    const int k_ins_max = 32;
    const int n_update = 5;
    const int seed = 100;

    std::vector<double> E(n_sites);
    boost::multi_array<T,2> phase(boost::extents[n_sites][n_sites]);

    for (int i=0; i<n_sites; ++i) {
        E[i] = (double) i;
        //std::cout << phase[i] << std::endl;
    }
    for (int i=0; i<n_sites; ++i) {
        for (int j=i; j<n_sites; ++j) {
            phase[i][j] = std::exp(std::complex<double>(0.0, 1.*i*(2*j+1.0)));
            phase[j][i] = myconj(phase[i][j]);
        }
    }

    general_U_matrix<T> Uijkl(n_sites, U, alpha);

    itime_vertex_container itime_vertices_init;
    itime_vertices_init.push_back(itime_vertex(0, 0, 0.5*beta, 2, true));

    // initialize submatrix_update
//SubmatrixUpdate<T> submatrix_update(k_ins_max, n_flavors, DiagonalG0<T>(beta), &Uijkl, beta, itime_vertices_init);
SubmatrixUpdate<T> submatrix_update(k_ins_max, n_flavors, OffDiagonalG0<T>(beta, n_sites, E, phase), &Uijkl, beta, itime_vertices_init);

submatrix_update.sanity_check();

// init udpate_manager
alps::params params;
params["BETA"] = beta;
params["FLAVORS"] = n_flavors;
params["N_MULTI_VERTEX_UPDATE"] = Nv_max;
params["DOUBLE_VERTEX_UPDATE_A"] = 1.0/beta;
params["DOUBLE_VERTEX_UPDATE_B"] = 1.0e-2;
VertexUpdateManager<T> manager(params, Uijkl, OffDiagonalG0<T>(beta, n_sites, E, phase), false);

// initialize RND generator
//std::vector<double> probs(Nv_max, 1.0);
boost::random::uniform_smallint<> dist(1,Nv_max);
boost::random::uniform_01<> dist01;
boost::random::mt19937 gen(seed);
boost::random::variate_generator<boost::random::mt19937&, boost::random::uniform_smallint<> > Nv_prob(gen, dist);
boost::random::variate_generator<boost::random::mt19937&, boost::random::uniform_01<> > random01(gen, dist01);

std::vector<alps::numeric::matrix<T> > M(n_flavors), M_scratch(n_flavors);


for (int i_update=0; i_update<n_update; ++i_update) {
T sign_from_M0, weight_from_M0;
boost::tie(sign_from_M0,weight_from_M0) = submatrix_update.compute_M_from_scratch(M_scratch);

//const T weight_rat = submatrix_update.vertex_insertion_removal_update(manager, random01);
const T weight_rat = manager.do_ins_rem_update(submatrix_update, Uijkl, random01, 1.0);
const T sign_bak = submatrix_update.sign();

ASSERT_TRUE(submatrix_update.sanity_check());
submatrix_update.recompute_matrix(true);
submatrix_update.compute_M(M);
T sign_from_M, weight_from_M;
boost::tie(sign_from_M,weight_from_M) = submatrix_update.compute_M_from_scratch(M_scratch);

ASSERT_TRUE(my_equal(weight_from_M/weight_from_M0, weight_rat, 1E-5));

//std::cout << "sign " << sign_bak << " " << submatrix_update.sign() << std::endl;
//std::cout << "sign_from_M " << sign_from_M << std::endl;
ASSERT_TRUE(std::abs(sign_bak-submatrix_update.sign())<1.0e-5);
ASSERT_TRUE(std::abs(sign_from_M-submatrix_update.sign())<1.0e-5);
//std::cout << " Nv " << submatrix_update.itime_vertices().num_interacting() << std::endl;
for (int flavor=0; flavor<n_flavors; ++flavor) {
if (M[flavor].num_cols()>0) {
ASSERT_TRUE(alps::numeric::norm_square(M[flavor]-M_scratch[flavor])/alps::numeric::norm_square(M[flavor])<1E-8);
}
}

const T weight_rat2 = manager.do_spin_flip_update(submatrix_update, Uijkl, random01);

T sign_from_M2, weight_from_M2;
boost::tie(sign_from_M2,weight_from_M2) = submatrix_update.compute_M_from_scratch(M_scratch);
ASSERT_TRUE(my_equal(weight_from_M2/weight_from_M, weight_rat2, 1E-5));

const T weight_rat3 = manager.do_shift_update(submatrix_update, Uijkl, random01, false);
}

//std::cout << DiagonalG0<T>(beta)(0.0) << std::endl;
//std::cout << DiagonalG0<T>(beta)(0.5*beta) << std::endl;
//std::cout << DiagonalG0<T>(beta)(0.9999*beta) << std::endl;
}
*/
