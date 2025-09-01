module LinearOptimizationModule

using DynamicExpressions: AbstractExpression, AbstractExpressionNode, Node, get_tree, get_child, eval_tree_array
using LinearAlgebra: qr, \, norm
using Statistics: cor
using ..CoreModule: AbstractOptions, Dataset, DATA_TYPE, LOSS_TYPE
using ..PopMemberModule: PopMember
using ..LossFunctionsModule: eval_loss, loss_to_cost
using ..UtilsModule: get_birth_order
using ..ComplexityModule: compute_complexity

# 线性项结构
# LinearTerm 结构体定义
struct LinearTerm{T}
    coefficient::T
    expression::Node{T}
    importance::T
end

# 线性组合结构
struct LinearCombination{T<:DATA_TYPE}
    terms::Vector{LinearTerm{T}}
    constant::T
end

"""
检测表达式是否为线性组合形式 y = w1*A + w2*B + w3*C + constant
"""
function detect_linear_combination(tree::AbstractExpressionNode{T}, options::AbstractOptions) where T
    terms = LinearTerm{T}[]
    constant = zero(T)
    
    if !_extract_linear_terms!(tree, terms, constant, one(T), options)
        return nothing  # 不是线性组合
    end
    
    return LinearCombination(terms, constant)
end

# 递归提取线性项
function _extract_linear_terms!(node::AbstractExpressionNode{T}, terms::Vector{LinearTerm{T}}, 
                                constant::T, coeff::T, options::AbstractOptions) where T
    if node.degree == 0
        # 叶子节点
        if node.constant
            constant += coeff * node.val
        else
            # 变量项
            push!(terms, LinearTerm(coeff, node, abs(coeff)))
        end
        return true
    elseif node.degree == 1
        # 一元操作符 - 检查是否为负号
        op_idx = node.op
        operators = options.operators
        if op_idx <= length(operators.unaops) && operators.unaops[op_idx] == (-)
            return _extract_linear_terms!(get_child(node, 1), terms, constant, -coeff, options)
        else
            # 其他一元操作符作为复合项处理
            push!(terms, LinearTerm(coeff, node, abs(coeff)))
            return true
        end
    elseif node.degree == 2
        # 二元操作符
        op_idx = node.op
        operators = options.operators
        left = get_child(node, 1)
        right = get_child(node, 2)
        
        if op_idx <= length(operators.binops) && operators.binops[op_idx] == (+)
            # 加法：递归处理两个子树
            return _extract_linear_terms!(left, terms, constant, coeff, options) &&
                   _extract_linear_terms!(right, terms, constant, coeff, options)
        elseif op_idx <= length(operators.binops) && operators.binops[op_idx] == (*)
            # 乘法：检查是否为 constant * expression 或 expression * constant
            if left.degree == 0 && left.constant
                return _extract_linear_terms!(right, terms, constant, coeff * left.val, options)
            elseif right.degree == 0 && right.constant
                return _extract_linear_terms!(left, terms, constant, coeff * right.val, options)
            else
                # 复杂乘法项作为整体处理
                push!(terms, LinearTerm(coeff, node, abs(coeff)))
                return true
            end
        elseif op_idx <= length(operators.ops[2]) && operators.ops[2][op_idx] == (-)
            # 减法：左边正系数，右边负系数
            return _extract_linear_terms!(left, terms, constant, coeff, options) &&
                   _extract_linear_terms!(right, terms, constant, -coeff, options)
        else
            # 其他操作符作为复合项处理
            push!(terms, LinearTerm(coeff, node, abs(coeff)))
            return true
        end
    else
        # 更高阶操作符作为复合项处理
        push!(terms, LinearTerm(coeff, node, abs(coeff)))
        return true
    end
end

"""
计算线性项的重要程度，基于系数大小和表达式复杂度
"""
function calculate_importance(term::LinearTerm{T}, options::AbstractOptions) where T
    # 基于系数绝对值的重要性
    coeff_importance = abs(term.coefficient)
    
    # 基于表达式复杂度的重要性（复杂度越低越重要）
    complexity = compute_complexity(term.expression, options)
    complexity_importance = T(1) / (T(1) + T(complexity))
    
    # 综合重要性评分
    return coeff_importance * complexity_importance
end

"""
更新线性组合中所有项的重要程度
"""
function update_importance!(combination::LinearCombination{T}, options::AbstractOptions) where T
    for i in 1:length(combination.terms)
        term = combination.terms[i]
        new_importance = calculate_importance(term, options)
        combination.terms[i] = LinearTerm(term.coefficient, term.expression, new_importance)
    end
end

"""
检查线性项之间的重复性，使用相关系数作为膨胀参数
"""
function check_linear_redundancy(combination::LinearCombination{T}, dataset::Dataset{T,L}, 
                                options::AbstractOptions; correlation_threshold::T = T(0.95)) where {T,L}
    terms = combination.terms
    n_terms = length(terms)
    n_terms <= 1 && return Int[]
    
    # 更新重要程度
    update_importance!(combination, options)
    
    # 计算每个项在数据集上的输出
    outputs = Matrix{T}(undef, size(dataset.X, 2), n_terms)
    
    for (i, term) in enumerate(terms)
        output, success = eval_tree_array(term.expression, dataset.X, options)
        if !success
            return Int[]  # 评估失败，不进行优化
        end
        outputs[:, i] = output
    end
    
    # 计算相关矩阵并识别冗余项
    redundant_indices = Int[]
    for i in 1:n_terms
        for j in (i+1):n_terms
            correlation = abs(cor(outputs[:, i], outputs[:, j]))
            if correlation > correlation_threshold
                # 保留重要程度更高的项
                if terms[i].importance >= terms[j].importance
                    push!(redundant_indices, j)
                else
                    push!(redundant_indices, i)
                end
            end
        end
    end
    
    return unique(redundant_indices)
end

"""
使用最小二乘法优化线性组合的系数
"""
function optimize_linear_coefficients(combination::LinearCombination{T}, dataset::Dataset{T,L}, 
                                    options::AbstractOptions, redundant_indices::Vector{Int}) where {T,L}
    # 移除冗余项
    active_terms = [term for (i, term) in enumerate(combination.terms) if i ∉ redundant_indices]
    isempty(active_terms) && return nothing
    
    n_terms = length(active_terms)
    n_samples = size(dataset.X, 2)
    
    # 构建设计矩阵
    A = Matrix{T}(undef, n_samples, n_terms + 1)  # +1 for constant term
    
    for (i, term) in enumerate(active_terms)
        output, success = eval_tree_array(term.expression, dataset.X, options)
        if !success
            return nothing
        end
        A[:, i] = output
    end
    A[:, end] .= one(T)  # 常数项
    
    # 最小二乘求解
    try
        coeffs = A \ dataset.y
        
        # 更新系数
        for (i, coeff) in enumerate(coeffs[1:n_terms])
            active_terms[i] = LinearTerm(coeff, active_terms[i].expression, abs(coeff))
        end
        
        return LinearCombination(active_terms, coeffs[end])
    catch
        return nothing  # 求解失败
    end
end

"""
将优化后的线性组合转换回表达式树
"""
function linear_combination_to_tree(combination::LinearCombination{T}, options::AbstractOptions) where T
    terms = combination.terms
    isempty(terms) && return Node{T}(; val=combination.constant)
    
    # 获取操作符索引
    operators = options.operators
    add_op_idx = findfirst(op -> op == (+), operators.binops)
    mul_op_idx = findfirst(op -> op == (*), operators.binops)
    
    add_op_idx === nothing && error("Addition operator not found")
    mul_op_idx === nothing && error("Multiplication operator not found")
    
    # 构建加法树
    result = nothing
    
    for term in terms
        term_node = if abs(term.coefficient - one(T)) < eps(T)
            term.expression
        else
            # coefficient * expression
            coeff_node = Node{T}(; val=term.coefficient)
            Node{T}(mul_op_idx, coeff_node, term.expression)
        end
        
        if result === nothing
            result = term_node
        else
            result = Node{T}(add_op_idx, result, term_node)
        end
    end
    
    # 添加常数项
    if abs(combination.constant) > eps(T)
        constant_node = Node{T}(; val=combination.constant)
        result = Node{T}(add_op_idx, result, constant_node)
    end
    
    return result
end

"""
对个体进行多重线性性优化
"""
function optimize_multilinear_individual(member::PopMember{T,L}, dataset::Dataset{T,L}, 
                                       options::AbstractOptions) where {T,L}
    tree = get_tree(member.tree)
    
    # 检测线性组合
    combination = detect_linear_combination(tree, options)
    combination === nothing && return member, 0.0
    
    # 检查重复性
    redundant_indices = check_linear_redundancy(combination, dataset, options)
    
    # 优化系数
    optimized_combination = optimize_linear_coefficients(combination, dataset, options, redundant_indices)
    optimized_combination === nothing && return member, 0.0
    
    # 转换回表达式树
    new_tree = linear_combination_to_tree(optimized_combination, options)
    
    # 创建新的个体
    new_member = PopMember(
        dataset,
        new_tree,
        options;
        parent=member.ref,
        deterministic=options.deterministic
    )
    
    # 如果新个体更好，返回新个体
    if new_member.cost < member.cost
        return new_member, 1.0  # 1次评估
    else
        return member, 1.0
    end
end

"""
剔除重要程度最低的项（当线性组合过于复杂时）
"""
function remove_least_important_terms(combination::LinearCombination{T}, 
                                     max_terms::Int = 5) where T
    n_terms = length(combination.terms)
    n_terms <= max_terms && return combination
    
    # 按重要程度排序
    sorted_indices = sortperm(combination.terms, by=term -> term.importance, rev=true)
    
    # 保留前max_terms个最重要的项
    kept_terms = combination.terms[sorted_indices[1:max_terms]]
    
    return LinearCombination(kept_terms, combination.constant)
end
end  # module

function linear_combination_to_tree(
    coefficients::Vector{T}, features::Vector{Node{T}}, operators
) where {T}
    # 构建线性组合的树结构
    if length(coefficients) == 1
        # 单项：coefficient * feature
        return Node{T}(
            op=operators.ops[2][1],  # 乘法操作符
            l=Node{T}(val=coefficients[1]),
            r=features[1]
        )
    end
    
    # 多项：递归构建加法树
    tree = Node{T}(
        op=operators.ops[2][1],  # 乘法操作符
        l=Node{T}(val=coefficients[1]),
        r=features[1]
    )
    
    for i in 2:length(coefficients)
        term = Node{T}(
            op=operators.ops[2][1],  # 乘法操作符
            l=Node{T}(val=coefficients[i]),
            r=features[i]
        )
        tree = Node{T}(
            op=operators.ops[2][2],  # 加法操作符
            l=tree,
            r=term
        )
    end
    
    return tree
end