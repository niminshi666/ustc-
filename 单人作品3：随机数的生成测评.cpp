#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>


#define MAX_INTERVAL 10000

void count_frequency(){//count the frequency of random num(ues the way"x=rand()%n;")
	printf("请输入N：");
	int N,count_times;
	scanf("%d",&N);
	printf("请输入测评次数（生成随机数的数量)：");
	scanf("%d",&count_times); 
	printf("\n");
	printf("下面是生成的随机序列：\n");
	int times[N]={0};
	int i,x;
	srand(time(NULL));
	for(i=0;i<count_times;i++){
		x=rand()%N;printf("%d\t",x);
		times[x]++;
	}
    printf("\n");
	float p[N]={0};
	printf("统计每个数的出现概率p(i)结果如下：\n;");
	for(i=0;i<N;i++){
		p[i]=float(times[i])/float(count_times);printf("p(%d)=%f\t",i,p[i]);
	}
}

void count_repetition_intervals() {//统计间隔分布情况 
    int N, count_times;
    
    // 输入N和测试次数
    printf("请输入N的值(0到N-1的范围): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("错误: N必须是正整数!\n");
        return;
    }
    
    printf("请输入测试次数(随机数生成数量): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("错误: 测试次数必须是正整数!\n");
        return;
    }
    
    // 初始化间隔计数器
    int intervals[N][MAX_INTERVAL] = {0}; // intervals[i][k]表示数字i间隔k出现的次数
    int last_pos[N];                     // 记录每个数字上次出现的位置
    for (int i = 0; i < N; i++) {
        last_pos[i] = -1;                // 初始化为-1表示尚未出现
    }
    
    // 生成随机数并统计间隔
    srand(time(NULL));
    for (int i = 0; i < count_times; i++) {
        int x = rand() % N;printf("%d\t",x);
        
        // 如果不是第一次出现，记录间隔
        if (last_pos[x] != -1) {
            int gap = i - last_pos[x] - 1;
            if (gap < MAX_INTERVAL) {
                intervals[x][gap]++;
            }
        }
        
        // 更新上次出现位置
        last_pos[x] = i;
    }
    
    // 打印每个数字的间隔分布
    printf("\n=== 数字重复间隔分布统计 ===\n");
    printf("N = %d, 测试次数 = %d\n\n", N, count_times);
    
    for (int i = 0; i < N; i++) {
        printf("数字 %d 的间隔分布:\n", i);
        printf("间隔\t出现次数\t频率\n");
        
        int total_gaps = 0;
        for (int k = 0; k < MAX_INTERVAL; k++) {
            total_gaps += intervals[i][k];
        }
        
        // 只打印出现次数非零的间隔
        int printed = 0;
        for (int k = 0; k < MAX_INTERVAL; k++) {
            if (intervals[i][k] > 0) {
                double frequency = (double)intervals[i][k] / total_gaps;
                printf("%d\t%d\t\t%.6f\n", k, intervals[i][k], frequency);
                printed++;
                
                // 限制输出行数，避免过多
                if (printed >= 20 && k < MAX_INTERVAL - 1) {
                    printf("... (省略其他低频率间隔)\n");
                    break;
                }
            }
        }
        
        if (printed == 0) {
            printf("无重复出现的情况\n");
        }
        
        printf("\n");
    }
}
#define MAX_PRINT 100  // 最多打印的随机数数量

void entropy_test() {//熵值测试 
    int N, count_times;
    
    // 输入N和测试次数
    printf("请输入N的值(0到N-1的范围): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("错误: N必须是正整数!\n");
        return;
    }
    
    printf("请输入测评次数(生成随机数的数量): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("错误: 测评次数必须是正整数!\n");
        return;
    }
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    // 生成随机数并统计频率
    int samples[count_times];
    int counts[MAX_PRINT] = {0};  // 记录每个数值的出现次数
    
    printf("\n生成的随机数序列(最多显示前%d个):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        counts[samples[i]]++;
        
        // 打印前MAX_PRINT个随机数
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");  // 每行显示10个数
        } else if (i == MAX_PRINT) {
            printf("... (共%d个随机数)\n", count_times);
        }
    }
    
    // 计算实际熵值
    double entropy = 0.0;
    for (int i = 0; i < N; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / count_times;
            entropy -= p * log2(p);  // 信息熵公式
        }
    }
    
    // 计算期望熵值（均匀分布的熵）
    double ideal_entropy = log2(N);
    
    // 判断测试是否通过（设定阈值为理想熵的95%）
    double threshold = ideal_entropy * 0.95;
    int passed = (entropy >= threshold);
    
    // 打印测试结果
    printf("\n=== 熵测试结果 ===\n");
    printf("N = %d, 测评次数 = %d\n", N, count_times);
    printf("实际熵值: %.6f 比特/符号\n", entropy);
    printf("期望熵值: %.6f 比特/符号 (均匀分布的理论最大值)\n", ideal_entropy);
    printf("通过标准: 实际熵 ≥ 期望熵 × 95%%\n");
    printf("测试结果: %s\n", passed ? "通过" : "未通过");
    printf("熵效率: %.2f%%\n", (entropy / ideal_entropy) * 100);
}

void runs_test() {//游程测试 
    int N, count_times;
    
    // 输入N和测试次数
    printf("请输入N的值(0到N-1的范围): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("错误: N必须是正整数!\n");
        return;
    }
    
    printf("请输入测评次数(生成随机数的数量): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("错误: 测评次数必须是正整数!\n");
        return;
    }
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    // 生成随机数并转换为二进制序列（基于中位数）
    int samples[count_times];
    int binary[count_times];
    
    // 计算中位数（简化版：使用(N-1)/2作为近似中位数）
    double median = (N - 1) / 2.0;
    
    printf("\n生成的随机数序列(最多显示前%d个):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        binary[i] = (samples[i] >= median) ? 1 : 0;
        
        // 打印前MAX_PRINT个随机数
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");  // 每行显示10个数
        } else if (i == MAX_PRINT) {
            printf("... (共%d个随机数)\n", count_times);
        }
    }
    
    // 计算游程数（Runs）
    int runs = 1;
    for (int i = 1; i < count_times; i++) {
        if (binary[i] != binary[i-1]) {
            runs++;
        }
    }
    
    // 计算0和1的数量
    int n0 = 0, n1 = 0;
    for (int i = 0; i < count_times; i++) {
        if (binary[i] == 0) n0++;
        else n1++;
    }
    
    // 计算理论期望游程数和标准差
    double expected_runs = 2.0 * n0 * n1 / count_times + 0.5;
    double variance = 2.0 * n0 * n1 * (2.0 * n0 * n1 - count_times) / 
                     (count_times * count_times * (count_times - 1));
    double std_dev = sqrt(variance);
    
    // 计算z-score（标准正态分布下的统计量）
    double z_score = (runs - expected_runs) / std_dev;
    
    // 判断测试是否通过（95%置信区间：|z| < 1.96）
    int passed = (fabs(z_score) < 1.96);
    
    // 打印测试结果
    printf("\n=== 游程测试结果 ===\n");
    printf("N = %d, 测评次数 = %d\n", N, count_times);
    printf("中位数基准: %.1f\n", median);
    printf("0的数量: %d\n", n0);
    printf("1的数量: %d\n", n1);
    printf("实际游程数: %d\n", runs);
    printf("理论期望游程数: %.2f\n", expected_runs);
    printf("标准差: %.4f\n", std_dev);
    printf("z-score: %.4f\n", z_score);
    printf("通过标准: |z-score| < 1.96 (95%%置信区间)\n");
    printf("测试结果: %s\n", passed ? "通过" : "未通过");
}

void autocorrelation_test() {//自相关性测试 
    int N, count_times;
    
    // 输入N和测试次数
    printf("请输入N的值(0到N-1的范围): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("错误: N必须是正整数!\n");
        return;
    }
    
    printf("请输入测评次数(生成随机数的数量): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("错误: 测评次数必须是正整数!\n");
        return;
    }
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    // 生成随机数序列
    int samples[count_times];
    
    printf("\n生成的随机数序列(最多显示前%d个):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        
        // 打印前MAX_PRINT个随机数
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");  // 每行显示10个数
        } else if (i == MAX_PRINT) {
            printf("... (共%d个随机数)\n", count_times);
        }
    }
    
    // 计算序列均值
    double mean = 0.0;
    for (int i = 0; i < count_times; i++) {
        mean += samples[i];
    }
    mean /= count_times;
    
    // 计算自相关系数（滞后1）
    double numerator = 0.0;
    double denominator = 0.0;
    
    for (int i = 0; i < count_times - 1; i++) {
        numerator += (samples[i] - mean) * (samples[i+1] - mean);
        denominator += pow(samples[i] - mean, 2);
    }
    
    // 计算最后一个样本的分母部分
    denominator += pow(samples[count_times-1] - mean, 2);
    
    double autocorrelation = numerator / denominator;
    
    // 计算标准误差（理论值：对于随机序列，标准误差约为1/√n）
    double standard_error = 1.0 / sqrt(count_times);
    
    // 计算z-score（自相关系数与0的偏离程度）
    double z_score = autocorrelation / standard_error;
    
    // 判断测试是否通过（95%置信区间：|z| < 1.96）
    int passed = (fabs(z_score) < 1.96);
    
    // 打印测试结果
    printf("\n=== 自相关性测试结果 ===\n");
    printf("N = %d, 测评次数 = %d\n", N, count_times);
    printf("序列均值: %.4f\n", mean);
    printf("理论期望值: 0 (完全随机序列的自相关系数应接近0)\n");
    printf("滞后1的自相关系数: %.6f\n", autocorrelation);
    printf("标准误差: %.6f\n", standard_error);
    printf("z-score: %.4f\n", z_score);
    printf("通过标准: |z-score| < 1.96 (95%%置信区间)\n");
    printf("测试结果: %s\n", passed ? "通过" : "未通过");
}
double calculate_chi_square_p_value(double chi_square, int df);
void frequency_test() {
    int N, count_times;
    
    // 输入N和测试次数
    printf("请输入N的值(0到N-1的范围): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("错误: N必须是正整数!\n");
        while (getchar() != '\n'); // 清除输入缓冲区
        return;
    }
    
    printf("请输入测评次数(生成随机数的数量): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("错误: 测评次数必须是正整数!\n");
        while (getchar() != '\n'); // 清除输入缓冲区
        return;
    }
    
    // 动态分配内存
    int *samples = (int *)malloc(count_times * sizeof(int));
    int *counts = (int *)calloc(N, sizeof(int)); // 使用N作为大小
    
    if (!samples || !counts) {
        printf("内存分配失败!\n");
        if (samples) free(samples);
        return;
    }
    
    // 生成随机数并统计频率
    printf("\n生成的随机数序列(最多显示前%d个):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        counts[samples[i]]++;
        
        // 打印前MAX_PRINT个随机数
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");
        } else if (i == MAX_PRINT) {
            printf("... (共%d个随机数)\n", count_times);
        }
    }
    
    // 计算卡方统计量
    double expected = (double)count_times / N;
    double chi_square = 0.0;
    
    printf("\n=== 频率分布统计 ===\n");
    printf("数值\t实际次数\t期望次数\t偏差\n");
    for (int i = 0; i < N; i++) {
        double deviation = counts[i] - expected;
        chi_square += pow(deviation, 2) / expected;
        printf("%d\t%d\t\t%.2f\t\t%.2f\n", i, counts[i], expected, deviation);
    }
    
    // 计算自由度
    int degrees_of_freedom = N - 1;
    
    // 使用更准确的p值计算方法
    double p_value = calculate_chi_square_p_value(chi_square, degrees_of_freedom);
    
    // 判断测试是否通过
    int passed = (p_value > 0.05);
    
    // 打印测试结果
    printf("\n=== 频率分布测试结果 ===\n");
    printf("N = %d, 测评次数 = %d\n", N, count_times);
    printf("卡方统计量(Chi-Square): %.4f\n", chi_square);
    printf("自由度: %d\n", degrees_of_freedom);
    printf("p值: %.6f\n", p_value);
    printf("通过标准: p值 > 0.05 (分布与均匀假设无显著差异)\n");
    printf("测试结果: %s\n", passed ? "通过" : "未通过");
    
    // 释放内存
    free(samples);
    free(counts);
}

// 需要添加卡方分布p值计算函数
double calculate_chi_square_p_value(double chi_square, int df) {
    // 使用Gamma函数计算更准确的p值
    // 这里给出一个简化实现，实际应用中应该使用数学库
    if (df == 2) {
        return exp(-chi_square / 2);
    } else {
        // 简化近似，实际应用中应该使用更精确的计算方法
        double p = 0.5 * erfc(sqrt(chi_square / 2) - sqrt(df / 2 - 1));
        return p;
    }
}

void cumulative_sums_test() {//累加测试 
    int N, count_times;
    
    // 输入N和测试次数
    printf("请输入N的值(0到N-1的范围): ");
    if (scanf("%d", &N) != 1 || N <= 0) {
        printf("错误: N必须是正整数!\n");
        return;
    }
    
    printf("请输入测评次数(生成随机整数的数量): ");
    if (scanf("%d", &count_times) != 1 || count_times <= 0) {
        printf("错误: 测评次数必须是正整数!\n");
        return;
    }
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    // 生成随机数序列并转换为标准化值 (-1 或 +1)
    int samples[count_times];
    double normalized[count_times];
    
    printf("\n生成的随机数序列(最多显示前%d个):\n", MAX_PRINT);
    for (int i = 0; i < count_times; i++) {
        samples[i] = rand() % N;
        // 将随机数转换为 -1 或 +1 (基于是否大于等于中位数)
        normalized[i] = (samples[i] >= (N-1)/2.0) ? 1.0 : -1.0;
        
        // 打印前MAX_PRINT个随机数
        if (i < MAX_PRINT) {
            printf("%d ", samples[i]);
            if ((i + 1) % 10 == 0) printf("\n");  // 每行显示10个数
        } else if (i == MAX_PRINT) {
            printf("... (共%d个随机数)\n", count_times);
        }
    }
    
    // 计算正向累加和 (Cusum Forward)
    double max_forward = 0.0;
    double sum = 0.0;
    for (int i = 0; i < count_times; i++) {
        sum += normalized[i];
        if (fabs(sum) > max_forward) {
            max_forward = fabs(sum);
        }
    }
    
    // 计算反向累加和 (Cusum Reverse)
    double max_reverse = 0.0;
    sum = 0.0;
    for (int i = count_times-1; i >= 0; i--) {
        sum += normalized[i];
        if (fabs(sum) > max_reverse) {
            max_reverse = fabs(sum);
        }
    }
    
    // 计算统计量 (标准化后的最大累加和)
    double z_forward = max_forward / sqrt(count_times);
    double z_reverse = max_reverse / sqrt(count_times);
    
    // 计算p值 (使用近似公式)
    double p_forward = 1.0;
    double p_reverse = 1.0;
    
    // 对于z > 1.0，使用近似公式
    if (z_forward > 1.0) {
        p_forward = 2.0 * (1.0 - 0.5 * exp(-1.7725 * z_forward * z_forward));
    }
    
    if (z_reverse > 1.0) {
        p_reverse = 2.0 * (1.0 - 0.5 * exp(-1.7725 * z_reverse * z_reverse));
    }
    
    // 综合p值 (取最小值)
    double p_value = (p_forward < p_reverse) ? p_forward : p_reverse;
    
    // 判断测试是否通过 (p值大于0.01表示通过)
    int passed = (p_value > 0.01);
    
    // 打印测试结果
    printf("\n=== 累加和测试结果 ===\n");
    printf("N = %d, 测评次数 = %d\n", N, count_times);
    printf("正向最大累加和: %.6f\n", max_forward);
    printf("反向最大累加和: %.6f\n", max_reverse);
    printf("标准化统计量 (Z): 正向=%.6f, 反向=%.6f\n", z_forward, z_reverse);
    printf("p值: %.6f\n", p_value);
    printf("通过标准: p值 > 0.01 (无显著非随机性)\n");
    printf("测试结果: %s\n", passed ? "通过" : "未通过");
}

// 生成0到N-1之间的均匀分布随机浮点数
double uniform_random(double N) {
    return (double)rand() / RAND_MAX * N;
}

// 使用Box-Muller变换生成正态分布随机数
double normal_random(double mean, double stddev) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return mean + stddev * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// 计算数组的均值
double mean(double arr[], int n) {
    double sum = 0.0;
    int valid_count = 0;
    for (int i = 0; i < n; i++) {
        if (!isnan(arr[i]) && !isinf(arr[i])) {
            sum += arr[i];
            valid_count++;
        }
    }
    if (valid_count == 0) {
        return 0.0; // 避免除以零
    }
    return sum / valid_count;
}

// 计算数组的标准差
double stddev(double arr[], int n, double mean_val) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum_sq += (arr[i] - mean_val) * (arr[i] - mean_val);
    }
    return sqrt(sum_sq / (n - 1));
}

// 计算数组的中位数（会修改原数组）
double median(double arr[], int n) {
    // 冒泡排序
    double temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    
    if (n % 2 == 0)
        return (arr[n/2 - 1] + arr[n/2]) / 2.0;
    else
        return arr[n/2];
}

// Lilliefors检验的详细实现，返回检验统计量和p值
double lilliefors_test(double arr[], int n, double *p_value) {
    // 1. 复制并排序数据
    double sorted_data[10000];
    for (int i = 0; i < n; i++) {
        sorted_data[i] = arr[i];
    }
    
    // 冒泡排序
    double temp;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (sorted_data[j] > sorted_data[j + 1]) {
                temp = sorted_data[j];
                sorted_data[j] = sorted_data[j + 1];
                sorted_data[j + 1] = temp;
            }
        }
    }
double mean(double arr[], int n);
 
    // 2. 计算样本均值和标准差
    double sample_mean = mean(sorted_data, n);
    double sample_std = stddev(sorted_data, n, sample_mean);
    
    printf("\nLilliefors检验计算过程:\n");
    printf("------------------------\n");
    printf("样本均值: %.4f\n", sample_mean);
    printf("样本标准差: %.4f\n", sample_std);
    
    // 3. 计算经验累积分布函数(ECDF)和理论CDF之间的最大差异
    double max_diff = 0.0;
    int max_idx = 0;
    
    // 输出前10个点的计算过程
    printf("\n前10个数据点的ECDF与理论CDF比较:\n");
    printf("   数据点 | 标准化值 | 理论CDF   | ECDF_lower | ECDF_upper | 差异_lower | 差异_upper\n");
    printf("----------|----------|-----------|------------|------------|------------|------------\n");
    
    for (int i = 0; i < n; i++) {
        double z = (sorted_data[i] - sample_mean) / sample_std;
        // 标准正态分布的CDF的近似计算
        double phi = 0.5 * (1.0 + erf(z / sqrt(2.0)));
        
        double ecdf_lower = (double)i / n;
        double ecdf_upper = (double)(i + 1) / n;
        
        double diff_lower = fabs(phi - ecdf_lower);
        double diff_upper = fabs(phi - ecdf_upper);
        
        // 更新最大差异
        if (diff_lower > max_diff) {
            max_diff = diff_lower;
            max_idx = i;
        }
        if (diff_upper > max_diff) {
            max_diff = diff_upper;
            max_idx = i;
        }
        
        // 输出前10个点
        if (i < 10) {
            printf("%9d | %8.4f | %9.4f | %10.4f | %10.4f | %10.4f | %10.4f\n", 
                   i+1, z, phi, ecdf_lower, ecdf_upper, diff_lower, diff_upper);
        }
    }
    
    // 输出最大差异点
    double z_max = (sorted_data[max_idx] - sample_mean) / sample_std;
    double phi_max = 0.5 * (1.0 + erf(z_max / sqrt(2.0)));
    double ecdf_lower_max = (double)max_idx / n;
    double ecdf_upper_max = (double)(max_idx + 1) / n;
    
    printf("\n最大差异点: 数据点 #%d (值=%.4f, 标准化值=%.4f)\n", 
           max_idx+1, sorted_data[max_idx], z_max);
    printf("理论CDF: %.4f\n", phi_max);
    printf("经验CDF范围: [%.4f, %.4f]\n", ecdf_lower_max, ecdf_upper_max);
    printf("Lilliefors检验统计量 D = %.4f\n", max_diff);
    
    // 估计p值（简化方法）
    double critical_05 = 0.029 * sqrt(n); // 5%显著性水平临界值近似
    double critical_01 = 0.036 * sqrt(n); // 1%显著性水平临界值近似
    
    *p_value = (max_diff > critical_01) ? 0.01 : 
               (max_diff > critical_05) ? 0.05 : 
               0.10; // 简化估计
    
    printf("临界值 (5%%): %.4f\n", critical_05);
    printf("估计p值: %.4f\n", *p_value);
    
    return max_diff < critical_05;
}

// 均匀分布的卡方检验，返回详细计算过程
double chi_square_test(double arr[], int n, double lower, double upper, int bins) {
    // 初始化直方图计数
    int observed[100] = {0};
    
    // 计算理论期望频率
    double bin_width = (upper - lower) / bins;
    double expected = (double)n / bins;
    
    printf("\n均匀分布卡方检验计算过程:\n");
    printf("------------------------\n");
    printf("分箱数: %d\n", bins);
    printf("每箱理论频数: %.2f\n", expected);
    
    // 构建直方图
    for (int i = 0; i < n; i++) {
        int bin = (arr[i] - lower) / bin_width;
        if (bin < 0) bin = 0;
        if (bin >= bins) bin = bins - 1;
        observed[bin]++;
    }
    
    // 计算卡方统计量
    double chi_square = 0.0;
    
    // 输出前10个箱的计算过程
    printf("\n前10个箱的频数分布:\n");
    printf(" 箱号 | 区间范围       | 观察频数 | 期望频数 | (O-E)2/E\n");
    printf("------|----------------|----------|----------|---------\n");
    
    for (int i = 0; i < bins; i++) {
        double bin_start = lower + i * bin_width;
        double bin_end = bin_start + bin_width;
        double contribution = pow(observed[i] - expected, 2) / expected;
        chi_square += contribution;
        
        // 输出前10个箱
        if (i < 10) {
            printf("%5d | [%.2f, %.2f) | %8d | %8.2f | %7.4f\n", 
                   i+1, bin_start, bin_end, observed[i], expected, contribution);
        }
    }
    
    // 自由度 = bins - 1
    int df = bins - 1;
    
    // 卡方临界值（简化查表）
    double critical_05 = 0;
    if (df == 9) critical_05 = 16.919;  // 常见值
    else if (df == 19) critical_05 = 30.144;
    else critical_05 = df * 1.8;  // 近似值
    
    printf("\n卡方统计量: %.4f\n", chi_square);
    printf("自由度: %d\n", df);
    printf("临界值 (5%%): %.4f\n", critical_05);
    
    // 估计p值（简化方法）
    double p_value = (chi_square > critical_05) ? 0.05 : 0.10;
    printf("估计p值: %.4f\n", p_value);
    
    return chi_square < critical_05;
}

// 打印增强版文本直方图
void print_enhanced_histogram(double arr[], int n, int bins, const char* title) {
    // 找到数据范围
    double min = arr[0], max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }
    
    double range = max - min;
    double bin_width = range / bins;
    
    // 初始化直方图计数
    int hist[100] = {0};
    for (int i = 0; i < n; i++) {
        int bin = (arr[i] - min) / bin_width;
        if (bin >= bins) bin = bins - 1;
        hist[bin]++;
    }
    
    // 找到最大频数，用于归一化
    int max_count = 0;
    for (int i = 0; i < bins; i++) {
        if (hist[i] > max_count) max_count = hist[i];
    }
    
    // 打印直方图标题
    printf("\n%s (样本量=%d):\n", title, n);
    printf("范围: [%.2f, %.2f]\n", min, max);
    
    // 打印直方图
    for (int i = 0; i < bins; i++) {
        double bin_start = min + i * bin_width;
        double bin_end = bin_start + bin_width;
        
        // 计算星号数量（最大50个）
        int stars = hist[i] * 50 / max_count;
        
        // 格式化输出区间和频数
        printf("[%6.2f, %6.2f) | ", bin_start, bin_end);
        
        // 绘制星号
        for (int j = 0; j < stars; j++) {
            printf("*");
        }
        
        // 显示实际频数
        printf(" %d\n", hist[i]);
    }
}

void last_part_test() {
    double uniform_upper;  // 均匀分布的上限
    double normal_mean;    // 正态分布的均值
    double normal_stddev;  // 正态分布的标准差
    int count;             // 生成的随机数数量
    int bins = 20;         // 直方图分箱数
    double p_value;        // 用于存储检验p值
    
    // 初始化随机数种子
    srand(time(NULL));
    printf("随机数生成器测试程序\n");
    printf("====================\n");
    
    // 获取用户输入
    printf("请输入均匀分布的上限值 (例如10.0): ");
    scanf("%lf", &uniform_upper);
    
    printf("请输入要生成的随机数数量 (例如1000): ");
    scanf("%d", &count);
    
    printf("请输入正态分布的均值 (例如0.0): ");
    scanf("%lf", &normal_mean);
    
    printf("请输入正态分布的标准差 (例如25.0): ");
    scanf("%lf", &normal_stddev);
    
    // 验证输入
    if (count <= 0 || count > 10000) {
        printf("错误：随机数数量必须在1到10000之间\n");
        return ;
    }
    
    if (uniform_upper <= 0) {
        printf("错误：均匀分布上限必须大于0\n");
        return ;
    }
    
    if (normal_stddev <= 0) {
        printf("错误：正态分布标准差必须大于0\n");
        return ;
    }
    
    // 生成均匀分布随机数
    printf("\n=== 均匀分布随机数 (0到%.1f) ===\n", uniform_upper);
    double uniform[10000];
    
    for (int i = 0; i < count; i++) {
        uniform[i] = uniform_random(uniform_upper);
    }
    
    // 计算统计量
    double uniform_avg = mean(uniform, count);
    double uniform_std = stddev(uniform, count, uniform_avg);
    
    printf("生成了%d个0到%.1f之间的均匀分布随机数\n", count, uniform_upper);
    printf("平均值: %.4f (理论值: %.4f)\n", uniform_avg, uniform_upper/2.0);
    printf("标准差: %.4f (理论值: %.4f)\n", uniform_std, uniform_upper/sqrt(12));
    
    // 均匀分布卡方检验
    int is_uniform = chi_square_test(uniform, count, 0, uniform_upper, bins);
    printf("卡方检验结论: ");
    if (is_uniform) {
        printf("样本通过均匀性检验 (在5%%显著性水平下)\n");
    } else {
        printf("样本未通过均匀性检验，可能不是均匀分布\n");
    }
    
    // 打印均匀分布直方图
    print_enhanced_histogram(uniform, count, bins, "均匀分布直方图");
    
    // 生成正态分布随机数
    printf("\n=== 正态分布随机数 (均值%.1f，标准差%.1f) ===\n", normal_mean, normal_stddev);
    double normal[10000];
    
    for (int i = 0; i < count; i++) {
        normal[i] = normal_random(normal_mean, normal_stddev);
    }
    
    // 计算统计量
    double sample_mean = mean(normal, count);
    double sample_std = stddev(normal, count, sample_mean);
    double sample_median = median(normal, count);
    
    printf("生成了%d个正态分布随机数\n", count);
    printf("样本均值: %.4f (理论值: %.1f)\n", sample_mean, normal_mean);
    printf("样本标准差: %.4f (理论值: %.1f)\n", sample_std, normal_stddev);
    printf("样本中位数: %.4f (理论值: %.1f)\n", sample_median, normal_mean);
    
    // Lilliefors检验
    int is_normal = lilliefors_test(normal, count, &p_value);
    printf("Lilliefors检验结论: ");
    if (is_normal) {
        printf("样本通过正态性检验 (在5%%显著性水平下)\n");
    } else {
        printf("样本未通过正态性检验，可能不是正态分布\n");
    }
    
    // 打印正态分布直方图
    print_enhanced_histogram(normal, count, bins, "正态分布直方图");
    
    
}	

int main() { 
while(1){
	printf("请选择需要测试的统计模块：\n"); 
	printf("1-随机生成后统计每个数的出现概率\n");
	printf("2-随机生成后统计每个数重复出现的间隔的分布情况\n");
	printf("3-测评部分1：输入N和测评次数进行熵值测评\n") ;
	printf("4-测评部分2：输入N和测评次数进行游程测评\n");
	printf("5-测评部分3：输入N和测评次数进行自相关性测评\n");
	printf("6-测试部分4：输入N和测评次数进行频率测评\n");
	printf("7-测试部分5：输入N和测评次数进行累加测评\n");
	printf("8-测评“设计公式或函数，获得均匀分布的随机数，正态分布的随机数，并测试”模块\n"); 
	printf("9-退出\n");
	int choose=0;
	scanf("%d",&choose);
	switch (choose) {
    case 1:
        count_frequency();
        printf("\n");
        break;
    case 2:
	    count_repetition_intervals();
	    printf("\n");
        break;
    case 3:
        entropy_test();
        printf("\n");
        break;
    case 4:
	    runs_test();
	    printf("\n");
        break;
    case 5:
        autocorrelation_test();
        printf("\n");
        break;
    case 6:
    	frequency_test();
    	printf("\n");
        break;
    case 7:
        cumulative_sums_test(); 
        printf("\n");
        break;
    case 8:
	    last_part_test();
	    printf("\n");
        break;
    case 9:
	    return 0; 
    default:
        printf("请重新输入\n");
		printf("\n"); 
        break;
}
	
}  
    return 0;
}

