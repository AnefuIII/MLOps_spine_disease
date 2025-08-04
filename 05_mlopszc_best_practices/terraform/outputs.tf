# MLOps Spine Disease Infrastructure Outputs
# This file defines all the outputs from the infrastructure

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS cluster"
  value       = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.eks.cluster_oidc_issuer_url
}

output "cluster_oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if `enable_irsa = true`"
  value       = module.eks.cluster_oidc_provider_arn
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_iam_role_unique_id" {
  description = "Stable and unique string identifying the IAM role"
  value       = module.eks.cluster_iam_role_unique_id
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_primary_security_group_id" {
  description = "Cluster security group that was created by Amazon EKS for the cluster"
  value       = module.eks.cluster_primary_security_group_id
}

output "cluster_encryption_config" {
  description = "Cluster encryption configuration"
  value       = module.eks.cluster_encryption_config
}

output "cluster_platform_version" {
  description = "Platform version for the cluster"
  value       = module.eks.cluster_platform_version
}

output "cluster_status" {
  description = "Status of the EKS cluster"
  value       = module.eks.cluster_status
}

output "cluster_version" {
  description = "The Kubernetes version for the cluster"
  value       = module.eks.cluster_version
}

output "cluster_arn" {
  description = "The Amazon Resource Name (ARN) of the cluster"
  value       = module.eks.cluster_arn
}

output "cluster_id" {
  description = "The name/id of the EKS cluster"
  value       = module.eks.cluster_id
}

output "cluster_identity_providers" {
  description = "Map of cluster identity providers"
  value       = module.eks.cluster_identity_providers
}

output "cluster_addons" {
  description = "Map of cluster addon attributes"
  value       = module.eks.cluster_addons
}

output "cluster_cloudwatch_log_group_kms_key_id" {
  description = "The ARN of the KMS Key used to encrypt the CloudWatch Logs"
  value       = module.eks.cluster_cloudwatch_log_group_kms_key_id
}

output "cluster_cloudwatch_log_group_retention_in_days" {
  description = "Number of days to retain log events"
  value       = module.eks.cluster_cloudwatch_log_group_retention_in_days
}

output "cluster_cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch Log Group"
  value       = module.eks.cluster_cloudwatch_log_group_arn
}

output "cluster_fargate_profiles" {
  description = "Map of cluster Fargate profile attributes"
  value       = module.eks.cluster_fargate_profiles
}

output "cluster_node_groups" {
  description = "Map of cluster node groups"
  value       = module.eks.cluster_node_groups
}

output "cluster_node_groups_autoscaling_group_names" {
  description = "List of the autoscaling group names created by EKS node groups"
  value       = module.eks.cluster_node_groups_autoscaling_group_names
}

output "cluster_managed_node_groups" {
  description = "Map of cluster managed node groups"
  value       = module.eks.cluster_managed_node_groups
}

output "cluster_managed_node_groups_autoscaling_group_names" {
  description = "List of the autoscaling group names created by EKS managed node groups"
  value       = module.eks.cluster_managed_node_groups_autoscaling_group_names
}

output "cluster_self_managed_node_groups" {
  description = "Map of cluster self-managed node groups"
  value       = module.eks.cluster_self_managed_node_groups
}

output "cluster_self_managed_node_groups_autoscaling_group_names" {
  description = "List of the autoscaling group names created by EKS self-managed node groups"
  value       = module.eks.cluster_self_managed_node_groups_autoscaling_group_names
}

# VPC Outputs
output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_arn" {
  description = "The ARN of the VPC"
  value       = module.vpc.vpc_arn
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "private_subnet_arns" {
  description = "List of ARNs of private subnets"
  value       = module.vpc.private_subnet_arns
}

output "private_subnets_cidr_blocks" {
  description = "List of cidr_blocks of private subnets"
  value       = module.vpc.private_subnets_cidr_blocks
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "public_subnet_arns" {
  description = "List of ARNs of public subnets"
  value       = module.vpc.public_subnet_arns
}

output "public_subnets_cidr_blocks" {
  description = "List of cidr_blocks of public subnets"
  value       = module.vpc.public_subnets_cidr_blocks
}

output "nat_public_ips" {
  description = "List of public Elastic IPs created for NAT Gateway"
  value       = module.vpc.nat_public_ips
}

output "nat_gateway_ids" {
  description = "List of NAT Gateway IDs"
  value       = module.vpc.nat_gateway_ids
}

output "nat_instance_ids" {
  description = "List of NAT Instance IDs"
  value       = module.vpc.nat_instance_ids
}

output "nat_instance_public_ips" {
  description = "List of public Elastic IPs created for NAT instances"
  value       = module.vpc.nat_instance_public_ips
}

output "igw_id" {
  description = "The ID of the Internet Gateway"
  value       = module.vpc.igw_id
}

output "igw_arn" {
  description = "The ARN of the Internet Gateway"
  value       = module.vpc.igw_arn
}

output "egw_id" {
  description = "The ID of the Egress Only Internet Gateway"
  value       = module.vpc.egw_id
}

output "egw_arn" {
  description = "The ARN of the Egress Only Internet Gateway"
  value       = module.vpc.egw_arn
}

output "cgw_ids" {
  description = "List of Customer Gateway IDs"
  value       = module.vpc.cgw_ids
}

output "cgw_arns" {
  description = "List of Customer Gateway ARNs"
  value       = module.vpc.cgw_arns
}

output "this_customer_gateway" {
  description = "Map of Customer Gateway attributes"
  value       = module.vpc.this_customer_gateway
}

output "vgw_id" {
  description = "The ID of the VPN Gateway"
  value       = module.vpc.vgw_id
}

output "vgw_arn" {
  description = "The ARN of the VPN Gateway"
  value       = module.vpc.vgw_arn
}

output "default_vpc_id" {
  description = "The ID of the Default VPC"
  value       = module.vpc.default_vpc_id
}

output "default_vpc_arn" {
  description = "The ARN of the Default VPC"
  value       = module.vpc.default_vpc_arn
}

output "default_vpc_cidr_block" {
  description = "The CIDR block of the Default VPC"
  value       = module.vpc.default_vpc_cidr_block
}

output "default_vpc_default_security_group_id" {
  description = "The ID of the security group created by default on Default VPC creation"
  value       = module.vpc.default_vpc_default_security_group_id
}

output "default_vpc_default_network_acl_id" {
  description = "The ID of the default network ACL of the Default VPC"
  value       = module.vpc.default_vpc_default_network_acl_id
}

output "default_vpc_default_route_table_id" {
  description = "The ID of the default route table of the Default VPC"
  value       = module.vpc.default_vpc_default_route_table_id
}

output "default_vpc_instance_tenancy" {
  description = "Tenancy of instances spin up within Default VPC"
  value       = module.vpc.default_vpc_instance_tenancy
}

output "default_vpc_enable_dns_support" {
  description = "Whether or not the Default VPC has DNS support"
  value       = module.vpc.default_vpc_enable_dns_support
}

output "default_vpc_enable_dns_hostnames" {
  description = "Whether or not the Default VPC has DNS hostname support"
  value       = module.vpc.default_vpc_enable_dns_hostnames
}

output "default_vpc_main_route_table_id" {
  description = "The ID of the main route table associated with the Default VPC"
  value       = module.vpc.default_vpc_main_route_table_id
}

# ECR Outputs
output "ecr_repository_url" {
  description = "The URL of the ECR repository"
  value       = aws_ecr_repository.mlops_app.repository_url
}

output "ecr_repository_arn" {
  description = "The ARN of the ECR repository"
  value       = aws_ecr_repository.mlops_app.arn
}

output "ecr_repository_name" {
  description = "The name of the ECR repository"
  value       = aws_ecr_repository.mlops_app.name
}

# S3 Outputs
output "s3_bucket_id" {
  description = "The name of the S3 bucket"
  value       = aws_s3_bucket.ml_artifacts.id
}

output "s3_bucket_arn" {
  description = "The ARN of the S3 bucket"
  value       = aws_s3_bucket.ml_artifacts.arn
}

output "s3_bucket_region" {
  description = "The AWS region this bucket resides in"
  value       = aws_s3_bucket.ml_artifacts.region
}

# RDS Outputs
output "rds_instance_id" {
  description = "The RDS instance ID"
  value       = aws_db_instance.mlops.id
}

output "rds_instance_endpoint" {
  description = "The connection endpoint"
  value       = aws_db_instance.mlops.endpoint
}

output "rds_instance_arn" {
  description = "The ARN of the RDS instance"
  value       = aws_db_instance.mlops.arn
}

output "rds_instance_status" {
  description = "The RDS instance status"
  value       = aws_db_instance.mlops.status
}

# Load Balancer Outputs
output "alb_id" {
  description = "The ID and ARN of the load balancer"
  value       = aws_lb.mlops.id
}

output "alb_arn" {
  description = "The ARN of the load balancer"
  value       = aws_lb.mlops.arn
}

output "alb_dns_name" {
  description = "The DNS name of the load balancer"
  value       = aws_lb.mlops.dns_name
}

output "alb_zone_id" {
  description = "The canonical hosted zone ID of the load balancer"
  value       = aws_lb.mlops.zone_id
}

output "alb_target_group_arn" {
  description = "The ARN of the target group"
  value       = aws_lb_target_group.mlops.arn
}

output "alb_target_group_name" {
  description = "The name of the target group"
  value       = aws_lb_target_group.mlops.name
}

# CloudWatch Outputs
output "cloudwatch_log_group_name" {
  description = "The name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.mlops.name
}

output "cloudwatch_log_group_arn" {
  description = "The ARN of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.mlops.arn
}

# Security Group Outputs
output "alb_security_group_id" {
  description = "The ID of the ALB security group"
  value       = aws_security_group.alb.id
}

output "rds_security_group_id" {
  description = "The ID of the RDS security group"
  value       = aws_security_group.rds.id
}

# Connection Information
output "kubectl_config" {
  description = "kubectl config as generated by the module"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}

output "aws_auth_configmap_yaml" {
  description = "Formatted yaml output for base aws-auth configmap containing roles used in cluster node groups/fargate profiles"
  value       = module.eks.aws_auth_configmap_yaml
}

# Cost Estimation
output "estimated_monthly_cost" {
  description = "Estimated monthly cost for the infrastructure"
  value       = "This is an estimate and may vary based on actual usage"
}

# Health Check URLs
output "health_check_url" {
  description = "URL for the application health check"
  value       = "http://${aws_lb.mlops.dns_name}/health"
}

output "application_url" {
  description = "URL for the application"
  value       = "http://${aws_lb.mlops.dns_name}"
} 