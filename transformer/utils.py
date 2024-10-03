import torch
import torch.nn.functional as F

def unpatchify(x, patch_size, depth, height, width):
    """
    Reconstructs videos from patches without using F.fold.

    Args:
        x (torch.Tensor): Tensor of shape (bs, num_patches, D * H * W * in_channels)
        patch_size (tuple of int): Size of each patch as (D, H, W).
        depth (int): Original video depth (number of frames).
        height (int): Original video height.
        width (int): Original video width.

    Returns:
        torch.Tensor: Reconstructed video of shape (bs, depth, in_channels, height, width)
    """
    bs, num_patches, patch_dim = x.shape
    D, H, W = patch_size
    in_channels = patch_dim // (D * H * W)

    # Calculate the number of patches along each dimension
    num_patches_d = depth // D
    num_patches_h = height // H
    num_patches_w = width // W

    # Ensure num_patches equals num_patches_d * num_patches_h * num_patches_w
    assert num_patches == num_patches_d * num_patches_h * num_patches_w, "Mismatch in number of patches."

    # Reshape x to (bs, num_patches_d, num_patches_h, num_patches_w, D, H, W, in_channels)
    x = x.view(bs, num_patches_d, num_patches_h, num_patches_w, D, H, W, in_channels)

    # Permute x to (bs, num_patches_d, D, num_patches_h, H, num_patches_w, W, in_channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()

    # Reshape x to (bs, depth, height, width, in_channels)
    reconstructed = x.view(bs, depth, height, width, in_channels)

    # Permute back to (bs, depth, in_channels, height, width)
    reconstructed = reconstructed.permute(0, 1, 4, 2, 3).contiguous()

    return reconstructed

def strings_to_tensor(string_list):
    """
    Converts a list of strings, each representing a list (e.g., "[1, 2, 3]"), 
    into a PyTorch tensor.

    Args:
        string_list (list of str): A list of strings, where each string is a list in string form.

    Returns:
        torch.Tensor: A PyTorch tensor containing the data from the lists.
    """
    # Convert each string to a list using eval
    list_of_lists = [eval(s) for s in string_list]
    
    # Convert the list of lists to a PyTorch tensor
    tensor = torch.tensor(list_of_lists, dtype=torch.float32)
    
    return tensor

def random_mask(bs: int, depth: int, height: int, width: int, patch_size: tuple, mask_ratio: float) -> torch.Tensor:
    """
    Generates a random mask for patched videos. Randomly selects patches across depth, height, and width to mask.

    Args:
        bs (int): Batch size.
        depth (int): Depth of the video (number of frames).
        height (int): Height of the video.
        width (int): Width of the video.
        patch_size (tuple of int): Size of the patches as (D, H, W).
        mask_ratio (float): Ratio of patches to mask. Ranges from 0 to 1.

    Returns:
        mask (torch.Tensor): A tensor of shape (bs, num_patches) with values in {0, 1}.
    """
    D, H, W = patch_size
    num_patches_d = depth // D
    num_patches_h = height // H
    num_patches_w = width // W
    num_patches = num_patches_d * num_patches_h * num_patches_w
    num_patches_to_mask = int(num_patches * mask_ratio)
    
    # Create a tensor of random values
    rand_tensor = torch.rand(bs, num_patches)
    
    # Sort the random tensor and get the indices
    _, indices = torch.sort(rand_tensor, dim=1)
    
    # Create a mask tensor initialized with ones
    mask = torch.ones(bs, num_patches, device=rand_tensor.device)
    
    # Set the first num_patches_to_mask indices to 0 for each batch
    mask[torch.arange(bs).unsqueeze(1), indices[:, :num_patches_to_mask]] = 0
    
    return mask

def remove_masked_patches(patches, mask):
    """
    Removes the masked patches from the patches tensor while preserving batch dimensions.
    Returned tensor will have shape (bs, number_of_unmasked_patches, embed_dim).

    Args:
        patches (torch.Tensor): Tensor of shape (bs, num_patches, embed_dim)
        mask (torch.Tensor): Tensor of shape (bs, num_patches) with values {0,1}, where 0 indicates a masked patch.

    Returns:
        torch.Tensor: Tensor containing only the unmasked patches.
    """
    # Ensure mask is a boolean tensor
    mask = mask.bool()

    # Get batch size and embed dimension
    bs, num_patches, embed_dim = patches.shape

    # Expand mask to match the shape of patches for correct indexing
    mask = mask.unsqueeze(-1).expand(-1, -1, embed_dim)

    # Use masked_select and reshape to maintain batch size
    unmasked_patches = torch.masked_select(patches, mask).view(bs, -1, embed_dim)

    return unmasked_patches

def add_masked_patches(patches, mask):
    """
    Adds the masked patches to the patches tensor.
    Returned tensor will have shape (bs, num_patches, embed_dim).
    The missing patches will be filled with 0s.

    Args:
        patches (torch.Tensor): Tensor of shape (bs, number_of_unmasked_patches, embed_dim)
        mask (torch.Tensor): Tensor of shape (bs, num_patches) with values {0,1}, where 0 indicates a masked patch.

    Returns:
        torch.Tensor: Tensor with masked patches filled with zeros.
    """
    # Ensure mask is a boolean tensor
    mask = mask.bool()

    # Get the total number of patches and embed dimension
    bs, num_patches = mask.shape
    embed_dim = patches.shape[-1]

    # Create a tensor of zeros with the same shape and dtype as the patches tensor
    full_patches = torch.zeros(bs, num_patches, embed_dim, device=patches.device, dtype=patches.dtype)

    # Create a mask for where patches should be placed
    mask_indices = mask.nonzero(as_tuple=True)

    # Assign the unmasked patches back to their original positions
    full_patches[mask_indices[0], mask_indices[1]] = patches

    return full_patches